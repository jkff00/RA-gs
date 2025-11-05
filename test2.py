import os
import torch
from random import randint
import open3d as o3d
import numpy as np
import math

from tqdm import tqdm
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
#无深度
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix
from scene.dataset_readers import qvec2rotmat
import cv2
import json
from tqdm import tqdm
import torchvision
import torch.nn.functional as F

def print_mem(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] Alloc={allocated:.2f} MB, Reserved={reserved:.2f} MB")
#也许可以迁移到C++部分，用ros，只有渲染端单纯用于计算可见性和深度渲染，仅此而以
#但是考虑要语义后期，还是python比较好
# 按照 Driscoll–Healy 采样定理，SH=3，你代码里设置的角度量化分辨率应该是22.5度，对应半球的bin的数量为32个：
#理论上22.5度角分辨率应该有52个采样点，但是球谐基本上32个就理论收敛了
class PointState:
    def __init__(self, num_points, color_dim=3, angle_res = 22.5,device="cuda"):
        '''

        Args:
            num_points: 全局点数量
            color_dim: 颜色维度
            angle_res: 每个点对应角度观测的分辨率
            device:
        '''
        self.N = num_points
        self.C = color_dim
        self.device = device


        # --- 球面近似的参数 ---
        self.scale,self.R, self.capacity = self.angle_res_to_scale(angle_res)

        # --- 颜色统计 ---
        self.counts = torch.zeros(num_points, dtype=torch.int16, device=device)
        self.color_mean = torch.zeros(num_points, color_dim, device=device,dtype=torch.float16)
        self.color_M2 = torch.zeros(num_points, color_dim, color_dim, device=device,dtype=torch.float16)

        # --- 角度统计 (预先填充避免 ragged 存),sparse_voxel_grid style ---
        self.key2dir_table = self.init_key2dir_table().to(device=device, dtype=torch.float16)  # shape (R**3, 3)

        self.keys_exist = torch.full(
            (num_points,  self.capacity), -1, dtype=torch.int32, device=device
        )
        # 每个点当前写入位置 (N,)
        self.write_ptr = torch.zeros(num_points, dtype=torch.long, device=device)

        # --- 距离统计 ---
        self.min_dists = torch.full((num_points,), float("inf"), device=device,dtype=torch.float16)
    
    def add_new_points(self, new_num_points: int):
        """
        添加新点并扩展所有相关属性
        Args:
            new_num_points: 新增点的数量
        """
        # 扩展 counts, color_mean, color_M2, keys_exist 等属性
        self.counts = torch.cat([self.counts, torch.zeros(new_num_points, dtype=torch.int16, device=self.device)])
        self.color_mean = torch.cat([self.color_mean, torch.zeros(new_num_points, self.C, device=self.device, dtype=torch.float16)])
        self.color_M2 = torch.cat([self.color_M2, torch.zeros(new_num_points, self.C, self.C, device=self.device, dtype=torch.float16)])
        
        # 扩展角度相关统计信息
        self.keys_exist = torch.cat([self.keys_exist, torch.full((new_num_points, self.capacity), -1, dtype=torch.int32, device=self.device)], dim=0)
        self.write_ptr = torch.cat([self.write_ptr, torch.zeros(new_num_points, dtype=torch.long, device=self.device)], dim=0)
        
        # 扩展距离统计
        self.min_dists = torch.cat([self.min_dists, torch.full((new_num_points,), float("inf"), device=self.device, dtype=torch.float16)], dim=0)

# ============角度存量的更新===================#
    def angle_res_to_scale(self,angle_res_deg: float):
        """
        输入角分辨率(度)，输出 scale 和 R
        """
        angle_res_rad = math.radians(angle_res_deg)

        s = (2.0 * math.sin(angle_res_rad / 2.0)) / math.sqrt(3.0)# 体素边长
        scale = 1.0 / s #预先用于乘法
        R = math.ceil(2.0 * scale)  # 每维格子数
        capacity = int(1 / (1 - math.cos(angle_res_rad / 2)))#最大角度观测容量 球冠覆盖（半球）

        return scale, R,capacity

    def ijk_to_key(self,ijk: torch.Tensor, R: int) -> torch.Tensor:
        """
        (N,3) -> (N,) int32 key
        """
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        key = (i * (R * R) + j * R + k).to(torch.int32)
        return key

    def key_to_ijk(self,key: torch.Tensor, R: int) -> torch.Tensor:
        """
        (N,) -> (N,3) int32
        """
        i = torch.div(key, R * R, rounding_mode='floor')
        j = torch.div(key, R, rounding_mode='floor') % R
        k = key % R
        return torch.stack([i, j, k], dim=-1).to(torch.int32)

    def voxelize_rays(self,rays: torch.Tensor, R: int) -> torch.Tensor:
        """
        rays: (N,3) 已归一化射线方向 (float)
        R: 每维格子数
        返回: (N,3) int32 索引
        """
        # 把 [-1,1] 映射到 [0,R)
        ijk = torch.floor((rays + 1.0) * 0.5 * R).to(torch.int32)
        return ijk.clamp(0, R - 1)

    def build_plane_basis_from_q_auto(self,q, eps=1e-8):
        """
        q: (B,3) 任意向量（建议先归一化）
        返回: u, v 形状 (B,3)，满足 u·q = 0, v·q = 0, u·v = 0
        """
        dtype, device = q.dtype, q.device

        # 找最小绝对分量的索引
        ids = torch.abs(q).argmin(dim=1)

        # 直接索引预定义坐标轴，不用 one_hot
        axes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype, device=device)
        ref = axes[ids]  # (B,3)

        # 投影 ref 到 q 的切平面：a' = ref - (q·ref) q
        qdotref = (q * ref).sum(dim=1, keepdim=True)
        a_prime = ref - q * qdotref

        # 归一化并构造正交基
        a_norm = a_prime.norm(dim=1, keepdim=True).clamp_min(eps)
        u = a_prime / a_norm
        v = torch.stack([
            q[:, 1] * u[:, 2] - q[:, 2] * u[:, 1],
            q[:, 2] * u[:, 0] - q[:, 0] * u[:, 2],
            q[:, 0] * u[:, 1] - q[:, 1] * u[:, 0]
        ], dim=1)

        return u, v

    def init_key2dir_table(self, device="cuda", eps=1e-8):
        """
        在 PointState.__init__ 或初始化流程里调用一次，构造 dense table:
          self.key2dir_table: (R**3, 3) float tensor, 每个索引对应 key -> 单位方向向量
        参数:
          device: 放表的设备，默认 self.device
          dtype: float32 推荐
        """
        device = device if device is not None else getattr(self, "device", "cpu")
        R = int(self.R)
        Nkeys = R ** 3

        # safety: 如果太大给出警告 (可选)
        # 例如当表大于 1e8 条目（约 > 1.2GB）要谨慎
        est_bytes = Nkeys * 3 * 4  # float32 估算
        if est_bytes > 2_000_000_000:  # ~2GB 警告阈值，可按需调整
            print(f"Warning: building dense key2dir table with R={R} -> {Nkeys} entries (~{est_bytes / 1e9:.2f} GB).")

        # 1) 生成 keys 0..Nkeys-1
        keys = torch.arange(Nkeys, device=device, dtype=torch.long)

        # 2) 解码 i,j,k（与你的 key 编码保持一致）
        ijk = self.key_to_ijk(keys, self.R)
        # 3) 体素中心 -> 映射到 [-1,1]^3
        centers = (ijk + 0.5) / float(R - 1) * 2.0 - 1.0  # (Nkeys,3)

        # 4) 归一化为单位向量
        norms = centers.norm(dim=1, keepdim=True).clamp_min(eps)  # (Nkeys,1)
        dirs = centers / norms  # (Nkeys,3)
        return dirs
        # 5) 存到对象中（放在指定 device）

    # ========== 批量更新 ==========
    def update_color(self, idx_batch: torch.Tensor, color_batch: torch.Tensor):
        """
        idx_batch: (B,) long tensor
        color_batch: (B,C) float tensor
        """
        # 索引出要更新的点
        idx = idx_batch
        cnt = self.counts[idx] + 1  # (B,)
        delta = color_batch - self.color_mean[idx]  # (B,C)
        # 新的 mean
        new_mean = self.color_mean[idx] + delta / cnt.unsqueeze(-1)  # (B,C)
        # 更新 M2
        self.color_M2[idx] += torch.einsum("bi,bj->bij", color_batch - new_mean, delta)

        # 写回
        self.counts[idx] = cnt
        self.color_mean[idx] = new_mean

    def update_angle(self, idx_batch: torch.Tensor,new_rays: torch.Tensor):
        """
        keys_exist: (N,M，) 已存 key (int32)
        new_rays:   (N,3) 新射线 (float, 已归一化) 当前视点和每个点的射线，从三维点指向视点，视点-三维点
        R: 每维格子数

        返回:
          keys_new: (M+K,) 更新后的 keys（包含新增）
          keys_added: (K,) 这次新增的 keys
        """
        ijk_new = self.voxelize_rays(new_rays, self.R)  # (N,3)
        keys_new = self.ijk_to_key(ijk_new, self.R)  # (N,)

        # 判重：找出新key里不存在于已有keys的部分
        eq = (keys_new.unsqueeze(1) == self.keys_exist[idx_batch])  # (N,K)
        exists = eq.any(dim=1)  # (N,)
        mask = ~exists

        rows = idx_batch[mask]  # (M,)需要输入观测的点索引
        cols = self.write_ptr[idx_batch][mask]  # (M,）对应点索引输入观测的位置

        # clamp 到 [0,32]，32 是 dummy slot

        self.keys_exist.index_put_((rows, cols), keys_new[mask])#输入观测的key，
        self.write_ptr.index_put_((rows,), torch.clamp(cols+1, max=self.capacity - 1))#输入观测后位置后移
        # 限制最大为32，等于32个key + 1
        # dummy
        # slot
        # 所有越界的都放在这两



    def update_distance(self, idx_batch: torch.Tensor, dist_batch: torch.Tensor):
        """
        idx_batch: (B,) long tensor
        dist_batch: (B,) float tensor
        """
        self.min_dists[idx_batch] = torch.minimum(self.min_dists[idx_batch], dist_batch.half())




    def update_all(self, idx_batch: torch.Tensor, color_batch: torch.Tensor, new_rays: torch.Tensor, dist_batch: torch.Tensor):
        '''

        Args:
            idx_batch: GS返回的可见点索引
            color_batch: 对应的归一化颜色向量
            new_rays: 归一化每个点到视点的射线
            dist_batch: 归一化过程保存的每个点到视点的无偏距离

        Returns:

        '''
#是否需要吧角度判断引入到全局，
        self.update_color(idx_batch, color_batch)

        self.update_angle(idx_batch, new_rays)

        self.update_distance(idx_batch, dist_batch)

    def compute_final_score(self,color_dists: torch.Tensor,
                            angle_mins: torch.Tensor,
                            dist_scores: torch.Tensor,
                            counts: torch.Tensor):
        """
        根据 Where-to-Render 论文公式，计算最终每个点的可渲染性指标 F(v_t, p).
        约束：
          - 如果观测次数 < 2，则指标 = 0

        Args:
            color_dists: (B,) 颜色一致性 [0,1]，query() 返回
            angle_mins:  (B,) 角度余弦值，query() 返回
            dist_scores:(B,) 距离分数 [0,1]，query() 返回
            counts:     (B,) 每个点的观测次数 (来自 self.counts[idx_batch])

        Returns:
            final_scores: (B,) 每个点的综合可渲染性得分
        """
        # h_geo
        #在大部分场景中色差是很小的，除非光照的影响，这里不应该作为一个约束
        #如果色差很小比如0.01，此时指标就会变得很高，因为weight是接近0，所以需要进行改良，然后跑一下三个指标各自的表现
        #原来的方法就是认为着色部分，色差小就不需要很多源视图，而GS不一样，所以色差权重需要消除，此外，需要考虑这个指标的映射函数是exp还是线性
        #根据GS特性为每一个指标涉设计衰减函数
        # ，尽量与实际需求匹配


        h_geo = color_dists.clamp(0.0, 1.0)
        # h_ang = torch.exp(angle_mins-1)
        h_res = (1-dist_scores)**(1-angle_mins*h_geo)
        #就这个，因为说的通，接下来就看实验效果，地面上的效果差可以渲染全景看一下，是不是因为垂直地面的观测不到位导致那些尖刺影响
        #就是那些和渲染质量对不上的部分，是不是因为在画幅外的低质量点（直接渲染全景就可以知道了）
        #其实这里距离对于渲染质量的影响本来就很小，
        final_scores = h_geo ** (1-angle_mins) * angle_mins * h_res#修改衰减函数只能改变分布，不改变排序，所以其实无所谓
        #把这个公式好好画画，解释清楚，然后赶紧嵌入pipeline
        # 色差越小值越大
        # 视角应该使用更加敏感的函数，距离不用那么敏感，都试试看，确定最后的指标
        # 距离一般不怎么发挥作用，从数值上exp似乎更好一点，高的不会特别高，低的也不会特别低，sqrt就比较极端，区分度很大大
        #指数衰减在刚开始衰减快，后面慢，平方衰减，刚开始衰减慢后面快,避免小角度大惩罚

        #指数衰减权重作为乘子保证了，当观测角度一致时，不会由于高噪声导致病态，保证了局部的稳定性，并且当角度偏离时，噪声会增加这种采样误差，此外，距离也作为一个权重，当存在高频时，希望可以用高分辨率表示，如果纹理很弱，就无需高分辨率表示提高h_res
        #h_res 这里也一样，当观测噪声降低，对分辨率变化的需求也降低了，趋向0，否则趋向1，

        # final_scores =h_geo* h_ang * h_res#三个指标要改成两个，颜色作为权重，目的是让角度为0的时候，整体指标也为1，避免重复拍摄，或者控制采样点也行



        # 条件过滤：观测次数 < 2 → 置零 一次观测，颜色均值为0指标为1，角度和距离是可以计算的
        final_scores = torch.where(counts >= 2, final_scores, torch.zeros_like(final_scores))

        return final_scores


    # ========== 批量查询 ==========
    #现有方法是怎么在观测很少的时候提供决策指导
#没有历史观测的部分设置为0，并且渲染的时候初始化图像为1，这样没有点的部分都是1
    #可以改到渲染器部分
    #可视化这里的颜色和对应图像和渲染的残差，这叫做可视化对比，然后另一方面，还要对比具体的PSNR和这个指标，定性和定量，完成这个预实验
    def query(self, idx_batch: torch.Tensor,new_rays: torch.Tensor,dist_batch: torch.Tensor,physical_scale: float):#分析一下点状态使用cpu，numpy计算还是GPU好
        """
        idx_batch: (B,) long tensor
        返回:
          color_dists: (B,) tensor
            angle_mins: (B,) tensor
          min_dists: (B,) tensor
        """
        # ---------- 颜色 ----------
        denom = (self.counts[idx_batch] - 1).clamp_min(1).half().view(-1, 1, 1)  # (B,1,1)

        covs = self.color_M2[idx_batch] / denom  # (B,C,C)
        trs = covs.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)  # (B,)颜色到均值的平方距离均值，所以2*trs = 颜色的两两平方距离均值，和颜色距离的均值还不一样，取值范围为

        color_dists = 1 - 1.154 * torch.sqrt(trs)  # (B,)直接用trace近似，取值范围为根号（1.5）要除以它实现归一化，因为trace的取值上限是0.75，根号（2 trace）近似颜色距离均值 要归一化
        #可以只输出每个指标单独检查看一下，是不是有问题

        # ---------- 距离 ----------
        #当一个像素能表达很小的物理距离的时候，这个点的距离指标应该就等与1.
        #其实距离本身就是判断观测到的物理距离的比值，对哈，
        # 这样写论文更清楚

        min_dists = self.min_dists[idx_batch]  # (B,)

        #距离应该+个权重，近距离的比例和远距离不一样，可以按照最小距离和物理尺度给出的参考距离做一个对比，
        dist_scores = (1 - dist_batch / (min_dists + 1e-8)).clamp_min(0.0)#如果大于最小深度就认为分辨率贡献足够，设置为0，该指标就为1
        dist_bound = min_dists<physical_scale
        #指标如何作为指导，如何终止，才能确认指标的最终形态
        #可变形网格，每个网格作为最小视点单位，避免重复拍摄

        #加权部分,物理尺度表示表达使用一个像素表达给定尺寸的物理空间时需要的深度，小于该深度认为再靠近已经没有意义，用于构造物理边界，避免视向选择的时候出现问题
        dist_scores[dist_bound] = 0.0#比如TUM数据集，距离拍摄距离更近也没关系，反正效果上不会出现问题
        # ---------- 夹角 ----------
        #计算每个查询向量的切平面
        u, v = self.build_plane_basis_from_q_auto(new_rays)#0.1 ms 2*(B,3)

        #角度还有一个很重要的点就是内插和外插，外插表示查询位于最外层的位置，此时可能和最近角度夹角很小，但是拟合质量会比相同夹角的内插要差
        #对观测数量不敏感：一个观测点就能把 h_min 拉到很小，但一个点远不代表可稳定拟合。远距离的一个角度会影响结果，也许夹角要融合距离
        # Step 1: 当前射线体素化
        # ijk_new = self.voxelize_rays(new_rays, self.R)  # (B,3)
        # Step 2: 历史 keys -> ijk (B,M,3)
        exist_keys = self.keys_exist[idx_batch].long()  # (B,M)


        # mask 有效位置 (B,M)，True = 有效 key
        valid_mask = exist_keys >= 0
        #add
        #判断是否为外插
        #front_mask = cos_theta > 0.0#取在同侧的射线，获取不取也行，对于一个点，侧面观测哪一边都行，先看看

        ray_hist = self.key2dir_table[exist_keys]#(B,M,3)
        cos_theta = torch.bmm(ray_hist, new_rays.unsqueeze(-1)).squeeze(-1)
        cos_theta = torch.where(valid_mask, cos_theta, torch.full_like(cos_theta, -1.0))#无效位设置为-1
        min_cos_theta = cos_theta.max(dim=1).values.clamp_min(0.0)#(B,)

        # #waicha
        ucoords = torch.bmm(ray_hist, u.unsqueeze(-1)).squeeze(-1)  # 等价于 u·ray_hist
        vcoords = torch.bmm(ray_hist, v.unsqueeze(-1)).squeeze(-1)
        u_min = torch.where(valid_mask, ucoords, torch.full_like(ucoords, float('inf'))).min(dim=1).values
        u_max = torch.where(valid_mask, ucoords, torch.full_like(ucoords, float('-inf'))).max(dim=1).values
        v_min = torch.where(valid_mask, vcoords, torch.full_like(vcoords, float('inf'))).min(dim=1).values
        v_max = torch.where(valid_mask, vcoords, torch.full_like(vcoords, float('-inf'))).max(dim=1).values
        inside_mask = (u_min <= 0) & (0 <= u_max) & (v_min <= 0) & (0 <= v_max)#B,
        # 不在包围盒内的翻倍角度：cos(2θ) = alpha*min_cos_theta
        min_cos_theta = torch.where(inside_mask, min_cos_theta, min_cos_theta.square())#反方向的时候值就是小于0，对外插的点进行惩罚0
        #因为外差会导致颜色噪声的影响也增大，因此颜色也需要控制
        color_dists = torch.where(inside_mask, color_dists, color_dists.square())

        return self.compute_final_score(color_dists, min_cos_theta, dist_scores,self.counts[idx_batch])
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def pixel_to_ray_pose(u, v, H, W, c2w):#这一步应该在C++实现，最好把所有计算都放到C++，只利用渲染工具
    """
    给定全景图像像素，输出以射线方向为Z轴的位姿矩阵。

    Args:
        u (int or torch.Tensor): 像素水平坐标
        v (int or torch.Tensor): 像素垂直坐标
        H (int): 图像高度
        W (int): 图像宽度
        c2w (torch.Tensor): 4x4 相机到世界矩阵

    Returns:
        c2w_ray (torch.Tensor): 4x4 射线坐标系在世界坐标系下的位姿
    """
    theta = (u / W) * 2 * torch.pi - torch.pi
    phi = (v / H) * torch.pi - torch.pi / 2

    # 相机坐标系下的方向向量
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)
    z_axis = torch.stack([x, y, z]).float()
    z_axis = z_axis / z_axis.norm()

    # 生成任意正交参考 X 轴
    tmp = torch.tensor([0.0, 1.0, 0.0])
    if torch.abs(torch.dot(z_axis, tmp)) > 0.99:
        tmp = torch.tensor([1.0, 0.0, 0.0])

    x_axis = torch.cross(tmp, z_axis)
    x_axis = x_axis / x_axis.norm()
    y_axis = torch.cross(z_axis, x_axis)

    # 构建旋转矩阵
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # 3x3

    # 平移使用相机位置
    t = c2w[:3, 3]

    c2w_ray = torch.eye(4)
    c2w_ray[:3, :3] = R
    c2w_ray[:3, 3] = t

    return c2w_ray
#forward cu不用管，因为preprocess没有改动
def render_spherical_simple(xyz,opacity,scale,rot,camera_center,
                     override_color=None,w2c_vp_pose=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    w2c = w2c_vp_pose


    projectionMatrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=math.pi / 2, fovY=math.pi / 2).transpose(0, 1).cuda()
    full_proj_transform = (w2c.unsqueeze(0).bmm(projectionMatrix.unsqueeze(0))).squeeze(0)

    # Set up rasterization configuration
    tanfovx = math.tan((math.pi / 2) * 0.5)
    tanfovy = math.tan((math.pi / 2) * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=2048,
        image_width=4096,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        spherical=True,
        debug=False
    )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    cov3D_precomp = None
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    colors_precomp = override_color#输入颜色 0～1 N，3
    #这里原来计算了每个点的方向向量，此时计算个夹角不是很简单吗，然后就可以直接获得指标，把指标覆盖颜色，渲染就好了
    #光珊化渲染是毫秒级别
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp,
                                       opacities=opacity, scales=scale, rotations=rot,
                                       cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim,
        sorted=True, return_inverse=True, return_counts=True)
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse+decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    # index = index.sort().values
    return unique, inverse, counts, index

def render_simple(width,height,xyz, opacity, scale, rot, camera_center,
                override_color=None, w2c_vp_pose=None,tanfovx=None,tanfovy=None,
                full_proj_transform=None,bg=None,visible_thresh = None,render_only = False,spherical =  False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=w2c_vp_pose,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        spherical=spherical,
        debug=False,
        visible_thresh = visible_thresh#add jxf
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    cov3D_precomp = None
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    colors_precomp = override_color  # 输入颜色 0～1 N，3
    # 这里原来计算了每个点的方向向量，此时计算个夹角不是很简单吗，然后就可以直接获得指标，把指标覆盖颜色，渲染就好了
    # 光珊化渲染是毫秒级别
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
#pixel_contrib 表示每个像素对应的gs均值对应的像素坐标，
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra,contrib,pixel_contrib = rasterizer(means3D=means3D, means2D=means2D, shs=shs,
                                                        colors_precomp=colors_precomp,opacities=opacity,
                                                        scales=scale, rotations=rot,
                                                        cov3Ds_precomp=cov3D_precomp)#原版没有s


    #可是逐像素不就代表基元对应那个像素了吗，只要展开，每个像素就有对应的基元，只要大于0应该就可以
    pixel_contrib = pixel_contrib.long()
    x = pixel_contrib[:, 0]
    y = pixel_contrib[:, 1]



    # # 合法 mask：0 <= x < W, 0 <= y < H
    mask = (contrib > 0)#在图像的边缘处，基元也会产生投影，因此需要过滤掉，这些均值都不在画幅内,计算指标的时候无所谓
    if not render_only:
        mask = mask & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    #pixel_contrib表示每个像素对应的GS均值对应的像素坐标，后面我们取unique后只取其均值像素对应的深度和颜色
    # 过滤画幅外的点
    unique_contrib, _,_,index_contrib = unique(contrib[mask], dim=0)#得到每个GS以及对应在原序列的索引

    unique_pixel_contrib = pixel_contrib[mask][index_contrib]
    #rendered_image, radii, normals, depths
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "contrib":unique_contrib,
            "pixel_contrib":unique_pixel_contrib,
            "depth": rendered_depth}

def save_initialized_ply_compatible(path, xyz, color, opacities, scales, rots):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 转 numpy
    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)  # 没有法向量，置 0
    f_dc = color.detach().cpu().numpy()[:, None, :]  # (N, 1, 3)，仿照官方结构
    f_rest = np.zeros((xyz.shape[0], 0))  # 没有高阶 SH
    opacities = opacities.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rots = rots.detach().cpu().numpy()

    # ============= 按 construct_list_of_attributes 拼字段 =============
    attribute_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1] * f_dc.shape[2]):  # 这里=3
        attribute_names.append(f"f_dc_{i}")
    for i in range(f_rest.shape[1] * f_rest.shape[0] if f_rest.size > 0 else 0):
        attribute_names.append(f"f_rest_{i}")
    attribute_names.append("opacity")
    for i in range(scales.shape[1]):  # 3
        attribute_names.append(f"scale_{i}")
    for i in range(rots.shape[1]):  # 4
        attribute_names.append(f"rot_{i}")

    dtype_full = [(name, 'f4') for name in attribute_names]

    # 拼接数据 (保持和上面顺序一致)
    f_dc_flat = f_dc.reshape(f_dc.shape[0], -1)   # (N, 3)
    attributes = np.concatenate(
        (xyz,
         normals,
         f_dc_flat,
         f_rest if f_rest.size > 0 else np.zeros((xyz.shape[0], 0)),
         opacities,
         scales,
         rots),
        axis=1
    )

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"✅ Saved initialized Gaussian PLY to {path}")

def compute_w2c_matrices(camera_centers, R_inv):
    """
    计算多个相机中心的世界到相机变换矩阵。

    参数：
        camera_centers (np.ndarray): 形状为 (N, 3) 的相机中心数组。
        R_inv (np.ndarray): 3x3 的旋转矩阵。

    返回：
        np.ndarray: 形状为 (N, 4, 4) 的变换矩阵数组。
    """
    N = camera_centers.shape[0]
    w2c_matrices = np.zeros((N, 4, 4), dtype=np.float32)
    w2c_matrices[:, :3, :3] = R_inv  # 设置旋转部分
    w2c_matrices[:, 3, 3] = 1.0  # 设置齐次坐标的最后一项为 1

    # 计算平移部分
    for i in range(N):
        w2c_matrices[i, :3, 3] = -R_inv @ camera_centers[i] #w2c
        w2c_matrices[i] = w2c_matrices[i].T #这里是为了方便cuda计算，和GS一样处理


    return w2c_matrices

def read_extrinsics_text(file_path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    path = file_path+"sparse/0/images.txt"
    image_dir = file_path + "images"
    w2c_list = []
    centers_list = []

    imgs_list = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                rot = qvec2rotmat(qvec)

                # construct w2c homogeneous
                w2c = np.eye(4)
                w2c[:3, :3] = rot
                w2c[:3, 3] = tvec
                w2c = w2c.T #这里是为了方便cuda计算，和GS一样处理

                # camera center in world coords: C = -R^{-1} tvec = -rot.T @ tvec
                center = (-rot.T @ tvec)

                w2c_list.append(w2c)
                centers_list.append(center)

                # 读取图片
                img_path = os.path.join(image_dir, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
                if img is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
                imgs_list.append(img.astype(np.float16) / 255.0 )       # 归一化到0~1)  # 保证0~255





    return np.stack(w2c_list, axis=0),np.stack(centers_list, axis=0),np.stack(imgs_list, axis=0)

def load_bundle_out(path):
    poses = []
    centers = []

    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # 跳过头部两行
    n_cams = int(lines[1].split()[0])
    idx = 2
    fix = np.diag([1, -1, -1, 1])  # 翻转 Y、Z
    for i in range(n_cams):
        # 行1: f, k1, k2
        fx, k1, k2 = map(float, lines[idx].split()); idx += 1

        # 行2-4: R
        R = np.array([
            list(map(float, lines[idx].split())),
            list(map(float, lines[idx+1].split())),
            list(map(float, lines[idx+2].split()))
        ])
        idx += 3

        # 行5: t
        t = np.array(list(map(float, lines[idx].split()))).reshape(3,1)
        idx += 1

        # 相机中心 (世界坐标)
        C = -R.T @ t

        # C2W 矩阵
        W2C = np.eye(4)
        W2C[:3,:3] = R
        W2C[:3, 3] = t.ravel()
        W2C_fixed = fix @ W2C
        poses.append(W2C_fixed.T)
        centers.append(C.ravel())

    return np.stack(poses), np.stack(centers)

def jet_colormap(x: torch.Tensor):#指标越高颜色B->G->R
    """
    x: (N,) in [0,1]
    return: (N,3) RGB in [0,1]
    """#指标越高，颜色越红，如果要指标越高，颜色越蓝，就需要1-
    x = 1-x.clamp(0, 1).unsqueeze(1)  # (N,1)
    x4 = 4*x
    r = torch.clamp(1.5 - torch.abs(x4 - 3), 0, 1)
    g = torch.clamp(1.5 - torch.abs(x4 - 2), 0, 1)
    b = torch.clamp(1.5 - torch.abs(x4 - 1), 0, 1)

    return torch.cat([r,g,b], dim=1)  # (N,3)

def compute_residuals(file_path, invert=True):
    """
    计算两组图像的残差并保存 jet 伪彩色图
    Args:
        gt_folder: GT 图像文件夹
        render_folder: 渲染图像文件夹
        save_folder: 如果提供，将残差图保存到该文件夹
        invert: True=高值蓝色，False=高值红色
    """
    gt_folder = os.path.join(file_path, "gt")
    render_folder = os.path.join(file_path, "renders")
    save_folder  = os.path.join(file_path, "residuals")

    gt_files = sorted(os.listdir(gt_folder))
    render_files = sorted(os.listdir(render_folder))
    assert gt_files == render_files, "GT 和 Render 文件名不一致！"

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    for fname in tqdm(gt_files, desc="Residual Cal"):
        gt_path = os.path.join(gt_folder, fname)
        render_path = os.path.join(render_folder, fname)

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        render_img = cv2.imread(render_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        if gt_img.shape != render_img.shape:
            raise ValueError(f"Shape mismatch: {fname}, GT {gt_img.shape}, Render {render_img.shape}")

        # 单通道残差 (均值)
        residual = np.abs(gt_img - render_img).mean(axis=2)  # (H,W), [0,1]

        # invert控制红蓝反转
        x = torch.from_numpy(residual).float().clamp(0, 1)
        if invert:
            x = 1 - x
        x = x.unsqueeze(2)  # (H,W,1)

        r = torch.clamp(1.5 - torch.abs(4*x - 3), 0, 1)
        g = torch.clamp(1.5 - torch.abs(4*x - 2), 0, 1)
        b = torch.clamp(1.5 - torch.abs(4*x - 1), 0, 1)

        color_map = torch.cat([r,g,b], dim=2).numpy()
        color_map = (color_map * 255).astype(np.uint8)  # (H,W,3)

        if save_folder is not None:
            save_path = os.path.join(save_folder, fname)
            cv2.imwrite(save_path, color_map)

# compute_residuals("/home/xf/CVPR2026/360-gaussian-splatting/data/pretest/train/ours_30000")
# print("done")
# read_extrinsics_text("/home/xf/CVPR2026/our_activate-gs/pre_test/livingroom200cm_generate_evaluation/livingroom5449/")
pcd = o3d.io.read_point_cloud("/home/xf/CVPR2026/gaussian-splatting/data/living_room/sparse/0/points3D.ply")  # 或者 .pcd
# -------------------------------
# 2. 体素滤波
# -------------------------------
voxel_size = 0.05  # 根据需要设置体素大小
pcd_down = pcd.voxel_down_sample(voxel_size)#实际应该是一半
# -------------------------------
# 3. 保留颜色并归一化
# -------------------------------
# 点位置

xyz = np.asarray(pcd_down.points)  # shape: (N, 3)

# 点颜色
color = np.asarray(pcd_down.colors)  # shape: (N, 3)
# 如果颜色是 0-255，需要归一化到 0-1
if color.max() > 1.0:
    color = color / 255.0
#初始尺度，0.866*体素边长

# 转成 PyTorch tensor，增量传输到GPU，减少耗时
xyz_tensor = torch.from_numpy(xyz).float()    # shape: (100,3)
color_tensor = torch.from_numpy(color).float()  # shape: (100,3)
import time
t1 = time.time()
# 如果需要放到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t2= time.time()
#这两个数据最费时间，如果增量式应该可以实时，现在就只有实现指标计算的代码以及实时性测试，然后再就是候选视点生成和选择

xyz_tensor = xyz_tensor.to(device)
color_tensor = color_tensor.to(device)
print("curr time2: ", time.time()-t2)

print("curr time: ", time.time()-t1)
N = xyz_tensor.shape[0]  # 点数量
print("Num of points: ", N)
print_mem("before point_state")
point_state = PointState(N) #构建点的状态
print_mem("after point_state")
# 初始化尺度，每个点的尺度相同，最小外接园
scale_val = 0.866 * voxel_size #0.866 根号3*0.5
scales = torch.full((N, 3), scale_val, dtype=torch.float32, device='cuda')
rots = torch.zeros((N, 4), device="cuda")
rots[:, 0] = 1
opacities = torch.ones((N, 1), dtype=torch.float, device="cuda")

#球面
R = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0]
])   # rotation you fixed,确保全景图平视镜头
R_inv = R.T
#camera_param
width = 1600
height = 1080
Fx = 685.7142857142857
Fy = 685.7142857142857
#判断可见点的参数
visible_thresh = (Fx*scale_val)**2

#透视
Fovx =2*math.atan(width/(2*Fx))#除以焦距
Fovy = 2*math.atan(height/(2*Fy))

tanfovx = math.tan(Fovx * 0.5)
tanfovy = math.tan(Fovy * 0.5)

projectionMatrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=Fovx, fovY=Fovy).transpose(0, 1).cuda()

#球面 渲染的时候可以调整分辨率进一步加速
width_spherical = 1600
height_spherical = 1600

Fovx_spherical = math.pi / 2
Fovy_spherical = math.pi / 2
tanfovx_spherical = math.tan(Fovx_spherical * 0.5)
tanfovy_spherical = math.tan(Fovy_spherical * 0.5)
Fx_spherical = width_spherical/ (2.0 * tanfovx)

#全景渲染时的表面点
visible_thresh_spherical = (Fx_spherical*scale_val)**2
projectionMatrix_spherical = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=Fovx_spherical, fovY=Fovy_spherical).transpose(0, 1).cuda()

physical_scale = Fx*0.01/5#求将一个厘米的体素作为3个像素需要的深度
print("physical_scale: ",physical_scale)

#读取位姿数据
poses,centers,images = read_extrinsics_text("/home/xf/CVPR2026/gaussian-splatting/data/living_room/")
poses_tensor = torch.from_numpy(poses).float().cuda()
centers_tensor = torch.from_numpy(centers).float().cuda()

images_tensor = torch.from_numpy(images).half().cuda()#图片显存占大头，1张大概100M，其他最多几百MB
bg = torch.tensor([0,0,0.5], dtype=torch.float32, device="cuda")#BGR HIGH->LOW
# bg = torch.tensor([0.5,0,0], dtype=torch.float32, device="cuda")#rgb
for i in range(poses.shape[0]):

    full_proj_transform = (poses_tensor[i].unsqueeze(0).bmm(projectionMatrix.unsqueeze(0))).squeeze(0)
    # Set up rasterization configuration
    # save_initialized_ply_compatible("/home/xf/CVPR2026/gaussian-splatting/output/7e104568-2/point_cloud/iteration_7000/point_cloud.ply", xyz_tensor, color_tensor, opacities, scales, rots)
    out = render_simple(width,height,xyz_tensor,opacities,scales,rots,centers_tensor[i],color_tensor,poses_tensor[i],tanfovx,tanfovy,full_proj_transform,bg=bg,visible_thresh=visible_thresh)

    #拿到涉及的点后，需要计算每个点对应的颜色，每个点的与相机视点相减的向量，由视点减去每个点的坐标，每个点的深度，因此返回每个点
    #在C++部分算好就拿过来，渲染大概25帧，2048*4096，如果1024*2048，是70帧，再结合多线程，可以同时渲染4～6张，候选视点如果是10个的化，差不多10 ms处理完
    #渲染本身不费时间，只要位姿C++批量计算好，统一传递，这边渲染很快

    # camera_center = np.asarray([[10.0, 3.5, 1.5]])#N*3
    #
    # w2c = compute_w2c_matrices(camera_center, R_inv)
    # w2c_tensor = torch.from_numpy(w2c).float().cuda()#N*4*4
    # camera_center_tensor = torch.from_numpy(camera_center).float().cuda()

    # out = render_spherical_simple(xyz_tensor,opacities,scales,rots,camera_center_tensor[0],color_tensor,w2c_tensor[0])
    #无非就是增加点，然后更新点的颜色为指标，然后视点
    # 取出渲染结果 (形状: [H, W, 3])

    # rendered_image = out["render"].permute(1, 2, 0).detach().cpu().numpy()
    # depth = out["depth"].permute(1, 2, 0)
    hidden_points = out["contrib"].long()#N，每个可观测的GS其均值的索引
    pixel_points = out["pixel_contrib"]#N,2，每个观测到的GS其均值的2D坐标
    # colors = color_tensor[hidden_points.long()]        # N x 3
    # # 创建图像初始化为黑色
    # image = torch.ones((1080, 1600, 3), dtype=colors.dtype, device=colors.device)

    x = pixel_points[:, 0]
    y = pixel_points[:, 1]#这个可以把不是mask内的点过滤掉，加入有mask
    color_gt = images_tensor[i][y, x]
    ray_points = centers_tensor[i]-xyz_tensor[hidden_points]#从三维点指向视点
    # dist_points = (ray_points**2).sum(dim=1).sqrt()#计算距离
    dist_points = ray_points.norm(dim=1) + 1e-8
    normaliz_ray_points = ray_points / dist_points.unsqueeze(1)#计算归一化向量
    point_state.update_all(hidden_points,color_gt,normaliz_ray_points,dist_points)

    #测试一下时间

    #渲染器得到的坐标遵循openGL,坐标起点是左下角，是上下颠倒的，所以要减去1
    #前端用lio得到体素
    #获取图像的颜色，深度，以及三维点，然后计算三维点向量存入索引对应的位置作为指标
    #深度是源深度减去伪深度除以源深度，原版是无偏深度，也就是在计算三维点射线归一化的时候存储一下距离
    #球面投影应该无偏和有偏是一样的，所以球面投影无所谓，直接从深度图获取即可，反而pinhole必须要求无偏深度
    #颜色通过协方差存储，Welford在线算法
    #夹角就是向量相似度最小的，这个可以先归一化再存储，然后当前射线点乘所有射线，取最小的再求角度即可
    #射线定义为从点出发到图像，维护一个observation，存储每个图像的id和距离，
    #然后射线那边存储单纯的id列表，根据id获取图像center,N，3的历史观测
    #距离简化就是1-当前距离比上历史距离，要求距离最小就是希望当前距离比上历史距离的比值最大,如果大于1就截断为0，因此距离存储最小的历史观测即可
    #现在每个点存一个PointStats,看样子还是需要批量处理，外部存一个N，3表示历史视点位置，就是这个centers_tensor


#test query
# poses_query,centers_query,_ = read_extrinsics_text("/home/xf/CVPR2026/our_activate-gs/pre_test/livingroom200cm_generate_evaluation/livingroom5449/")
#========================================================
#先进行几何重建，然后执行光度重建，确保几何完整性，或者用RGBD去做先
#颜色指标要随着角度变化，避免相似视角不改变指标
#即时差一点也没是，终止条件定义好
#lidar需要先定义几何然后再主动渲染，还是先做RGBD吧，然后背景也用红色，表达完整性，然后区域外的部分就用mask掩盖调，反正先和那个activesplat比较
#这个工作只能投个A会了，tro感觉没有机会，后面看看怎么搞个tro的，那就来的快
#解决球面渲染，解决视向选择，然后调研并找到当前的终止拍摄方案，最后把规划步骤搞定（先用activesplat的规划器好了），然后做个实机实验
#为了扩展，point_state需要不断扩展，外部通过体素计算新点，新点拼接到原来但点后面，颜色赋值一下就可以了
#前端占用地图，然后这里就存占用中心就可以了
#========================================================

poses_query,centers_query = load_bundle_out("/home/xf/CVPR2026/360-gaussian-splatting/data/pre_test_path/camera.out")
poses_tensor = torch.from_numpy(poses_query).float().cuda()
centers_tensor = torch.from_numpy(centers_query).float().cuda()

rederabilities = []
per_view = {}
#17 ms/ view
#关闭可视化的话大概是8～10ms，可以通过采样点直接赋值给图像tensor，计算视向，节省时间
for i in tqdm(range(poses_tensor.shape[0]), desc="Rendering & Evaluating"):

    full_proj_transform = (poses_tensor[i].unsqueeze(0).bmm(projectionMatrix_spherical.unsqueeze(0))).squeeze(0)
    # Set up rasterization configuration
    # save_initialized_ply_compatible("/home/xf/CVPR2026/gaussian-splatting/output/7e104568-2/point_cloud/iteration_7000/point_cloud.ply", xyz_tensor, color_tensor, opacities, scales, rots)
    out = render_simple(width_spherical,height_spherical,xyz_tensor,opacities,scales,
                        rots,centers_tensor[i],color_tensor,
                        poses_tensor[i],tanfovx_spherical,
                        tanfovy_spherical,full_proj_transform,
                        bg=bg,visible_thresh=visible_thresh_spherical,render_only= True,spherical= True)


    hidden_points = out["contrib"].long()  # N，每个可观测的GS其均值的索引
    ray_points = centers_tensor[i] - xyz_tensor[hidden_points]  # 从三维点指向视点
    # dist_points = (ray_points ** 2).sum(dim=1).sqrt()  # 计算距离
    dist_points = ray_points.norm(dim=1) + 1e-8

    normaliz_ray_points = ray_points / dist_points.unsqueeze(1)  # 计算归一化


    metrics_points = point_state.query(hidden_points, normaliz_ray_points.half(), dist_points.half(),physical_scale)#N,
    # 平均分数
    mean_val = metrics_points.mean().item()
    rederabilities.append(mean_val)


    # 存 per-view
    fname = f"{i:05d}.png"
    per_view[fname] = mean_val
    color_tensor[hidden_points] = jet_colormap(metrics_points).float()
    #如果你要的是网格球体，推荐用 TRIANGLE_LIST，自己定义球体顶点和三角面片，并为每个顶点指定颜色。
    #ros 发布球面渲染指标和全景图两种

    out = render_simple(width_spherical,height_spherical,xyz_tensor[hidden_points],opacities[hidden_points],
                        scales[hidden_points],rots[hidden_points],centers_tensor[i],
                        color_tensor[hidden_points],poses_tensor[i],
                        tanfovx_spherical,tanfovy_spherical,full_proj_transform,bg,
                        visible_thresh=visible_thresh_spherical,render_only= True,spherical =  True)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz_tensor[hidden_points].detach().cpu().numpy())
    # pcd.colors = o3d.utility.Vector3dVector(color_tensor[hidden_points].detach().cpu().numpy())
    # o3d.io.write_point_cloud(os.path.join("/home/xf/CVPR2026/360-gaussian-splatting/data/pre_test_path/pretest_pcd", '{0:05d}'.format(i) + ".pcd"), pcd)
    rendered_image = out["render"].permute(1, 2, 0).detach().cpu().numpy()
    torchvision.utils.save_image(out["render"], os.path.join("/home/xf/CVPR2026/360-gaussian-splatting/data/pre_test_path/pretest_renderability_spherical", '{0:05d}'.format(i) + ".png"))
# #如果单个指标设计消除不了，那么就看全局指标对不对的上，然后就下一步好了/home/xf/CVPR2026/360-gaussian-splatting/data/pre_test_path/5449
# per_view_dict = {
#         "ours_30000": {
#             "RENDERABILITY": per_view
#         }
# }
# # 代码上传到仓库，保留一个pretest版本
#
# with open(os.path.join("/home/xf/CVPR2026/360-gaussian-splatting/data/pre_test_path", "per_view.json"), "w") as f:
#     json.dump(per_view_dict, f, indent=2)
#删除渲染时视锥外的基元，就是渲染的时候如果均值在视锥外，那么他的基元颜色透明度应该就=0，渲染就不会加上去，看看有没有效果
#指标和渲染指标统计对比
#然后指标图和渲染图以及参残差图对比
#然后再体现一个实时性就可以了

# open3d保存点云，也可以把点索引和颜色指标传给C++，前端几何部分就用C++
