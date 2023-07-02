import logging
import math
import os
import time
from math import atan, ceil, cos, floor, pi, sin, sqrt, tan
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from kornia import create_meshgrid
from torch.optim.optimizer import Optimizer
from torchvision import transforms


# 自定义优化器
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        # 阈值判定
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
         
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss

class FishEyeGenerator: 
    # dst_shape format: [rows, cols]
    def __init__(self, focal_len, dst_shape):
        # 300
        self.focal_len = focal_len
        # 输出的鱼眼图像的行和列大小
        self.dst_shape = dst_shape

        # self.focal_len * pi表示成像求直径
        # 等距投影模型中鱼眼图像中点到畸变中心的距离为 f*theta, theta为弧度
        # 当theta = pi/2时，表示鱼眼图像最边缘边界到畸变中心的距离，即成像球半径
        # self.ratio表示标准图像短边与成像球的比例
        self.ratio = min(self.dst_shape[0], self.dst_shape[1]) / (self.focal_len * np.pi)
        # 虚拟世界坐标系Z轴
        self.world_z = 500

        # 鱼眼图像转换后的有效边界
        square_r = (min(self.dst_shape[0],self.dst_shape[1])/2)**2
        cord_x, cord_y = np.meshgrid(np.arange(self.dst_shape[1]), np.arange(self.dst_shape[0]))
        radius = (cord_x - self.dst_shape[1]/2)**2 + (cord_y - self.dst_shape[0]/2)**2
        self.bad_radius = (radius >= square_r).reshape(-1)
        

    def _init_pin_matrix(self, src_shape):
        rows = src_shape[0]
        cols = src_shape[1]
        self._pin_matrix = \
            np.array([
                [self.world_z, 0, cols/2, 0],
                [0, self.world_z, rows/2, 0],
                [0, 0, 1, 0]])


    def _calc_cord_map(self, cv_img):
        self._init_pin_matrix(cv_img.shape)
        # 待转换图像的尺寸
        src_rows = cv_img.shape[0]
        src_cols = cv_img.shape[1]
        # 转换成鱼眼图像的尺寸
        dst_rows = self.dst_shape[0]
        dst_cols = self.dst_shape[1]

        # 生成目标图像的坐标矩阵[640, 640]
        cord_x, cord_y = np.meshgrid(np.arange(dst_cols), np.arange(dst_rows))        
        # 将坐标从像素坐标系转换成图像坐标系，平移到中心点
        cord = np.dstack((cord_x, cord_y)).astype(np.float64) - np.array([dst_cols / 2, dst_rows / 2])
        # 展成两行[640*640, 2]
        cord = cord.reshape(-1, 2)
        
        # 将目标图像坐标缩小到鱼眼圈中坐标
        cord = np.array(cord) / self.ratio

        # 论文公式1计算矩形图像转换到球形图像中的theta    
        radius_array = np.sqrt(np.square(cord[:, 0]) + np.square(cord[:, 1])) + 1e-10
        theta_array = radius_array / self.focal_len

        # 论文公式2计鱼眼图像坐标转换到世界坐标系下的坐标
        new_x_array = np.tan(theta_array) * cord[:, 0] / radius_array * self.focal_len
        new_y_array = np.tan(theta_array) * cord[:, 1] / radius_array * self.focal_len
        
        # 找到中心点的mask
        temp_index1 = radius_array == 0
        # 找到x=0的一行的mask, x轴
        temp_index2 = cord[:, 0] == 0
        # 找到y=0的一行的mask, y轴
        temp_index3 = cord[:, 1] == 0
        
        # 中心点的mask
        bad_x_index = temp_index1 | (temp_index2 & temp_index1)
        bad_y_index = temp_index1 | (temp_index3 & temp_index1)
        
        new_x_array[bad_x_index] = 0
        new_y_array[bad_y_index] = 0

        # [409600, 1]
        new_x_array = new_x_array.reshape((-1, 1))
        new_y_array = new_y_array.reshape((-1, 1))
        
        # [409600, 2]
        new_cord = np.hstack((new_x_array, new_y_array))
        # self._PARAM表示世界坐标系中的垂直距离Z，初始化500
        # [409600, 3]
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1)) * self.world_z))
        # [409600, 4]
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1))))

        # shape=(pix_num, 3)
        # 将世界坐标系坐标转换成针孔相机下坐标，对应公式4
        pin_image_cords = np.matmul(self._pin_matrix, new_cord.T).T
        self._map_cols = pin_image_cords[:, 0] / pin_image_cords[:, 2]
        self._map_rows = pin_image_cords[:, 1] / pin_image_cords[:, 2]

        # 坐标取整[409600]
        self._map_cols = self._map_cols.round().astype(int)
        self._map_rows = self._map_rows.round().astype(int)

        index1 = self._map_rows < 0
        index2 = self._map_rows >= src_rows
        index3 = self._map_cols < 0
        index4 = self._map_cols >= src_cols
        index5 = pin_image_cords[:, 2] <= 0

        # 超出边界处理
        bad_index = index1 | index2 | index3 | index4 | index5
        bad_index = bad_index | self.bad_radius
        self._map_cols[bad_index] = cv_img.shape[1]
        self._map_rows[bad_index] = 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_ray_directions(H, W, focal):
    # 网格形式的索引序列1xHxWx2，取索引第0个，变成[HxWx2]
    # 网格未进行归一化，索引数据是0-799
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    # 按照最后一个维度拆分，变成[HxW]
    i, j = grid.unbind(-1)

    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    # 方向中的三个存储数据：
    # 第一个是x方向，当前像素光线向量的x方向分量
    # 第二个是y方向，当前像素光线向量的y方向分量
    # 第三个是z方向，当z=1时，前面的tan(x) = x
    # 详见：https://zhuanlan.zhihu.com/p/495652881

    # 参数Y和参数Z前面有负号的原因：
    # 因为COLMAP采用的是opencv定义的相机坐标系统，
    # 其中x轴向右，y轴向下，z轴向内；而Nerf pytorch采用的是OpenGL定义的相机坐标系统，
    # 其中x轴向右，y轴向上，z轴向外。因此需要在y与z轴进行相反数转换。
    
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rays(directions, c2w, box=None, img_dim = False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # 矩阵乘法，将相机坐标系下的光线角度转换成世界坐标系下的光线角度
    # print(directions)
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)

    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    # 世界坐标系下所有光线的原点坐标
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    # 截取中间的一部分
    if box is not None:
        rays_d = rays_d[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
        rays_o = rays_o[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

    if not img_dim:
        # (H*W, 3)，通道1:光线向量x方向分量，通道2:光线向量y方向分量，通道3:光线向量z方向分量(设定为1，方便其他两通道表示)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def save_render_img(results, save_pth, img_h, img_w, cur_epoch, show_results, batch):
    # 如果渲染图片没到1张
    if len(show_results["coarse"]*batch) < (640*640):
        # 计数光线数量，保存第一张图片
        coarse_part = results['rgb_coarse'].cpu().detach()
        fine_part = results['rgb_fine'].cpu().detach()
        # 保存部分结果
        show_results["coarse"] += [coarse_part]
        show_results["fine"] += [fine_part]
    else:
        if show_results["already_save"]:
            pass
        else:
        # 合并结果   
            show_results["coarse"] = torch.cat(show_results["coarse"], 0)
            show_results["fine"] = torch.cat(show_results["fine"], 0)

            img_coarse = show_results["coarse"][:640*640]
            img_fine = show_results["fine"][:640*640]
            # 取出两种渲染结果图片
            img_coarse = img_coarse.view(img_h, img_w, 3)
            img_fine = img_fine.view(img_h, img_w, 3)
            # 转换格式
            img_coarse = img_coarse.permute(2, 0, 1) # (3, H, W)
            img_fine = img_fine.permute(2, 0, 1) # (3, H, W)
            # 图片保存路径
            img_coarse_pth = os.path.join(Path(save_pth), Path("coarse_epoch_" + str(cur_epoch) + ".png"))
            img_fine_pth = os.path.join(Path(save_pth), Path("fine_epoch_" + str(cur_epoch) + ".png"))
            # 转为RGB保存
            img_coarse = transforms.ToPILImage()(img_coarse).convert("RGB")
            img_fine = transforms.ToPILImage()(img_fine).convert("RGB")
            # 保存图片
            img_coarse.save(img_coarse_pth)
            img_fine.save(img_fine_pth)

            show_results["already_save"] = True

# 自己写的前项代码中变量名称发生变化
# 需要对官方训练出来的模型权重文件进行编辑，对应相应的权重信息
def rewrite_weights_file_ckpt(nerf_ckpt_dict):
    # print(nerf_ckpt_dict)
    # 获取权重信息
    state_dict = nerf_ckpt_dict["model"]
    new_state_dict = state_dict.copy()
    # 删除、重命名权重信息
    for key in state_dict:
        # 删除粗查询的权重
        if key.split(".")[0] != "nerf_fine":
            del(new_state_dict[key])
        # 重命名细粒度查询的相关信息
        else:
            # 先删除
            del(new_state_dict[key])
            # 再写入
            new_key_name = key.replace((key.split(".")[0] + "."), "")
            new_state_dict[new_key_name] = state_dict[key]

    return new_state_dict

def rewrite_weights_file_rays_ckpt(nerf_ckpt_dict):
    # print(nerf_ckpt_dict)
    # 获取权重信息
    state_dict = nerf_ckpt_dict["model"]
    new_state_dict = state_dict.copy()
    # 删除、重命名权重信息
    for key in state_dict:
        # 删除粗查询的权重
        if key.split(".")[0] != "nerf_rays":
            del(new_state_dict[key])
        # 重命名细粒度查询的相关信息
        else:
            # 先删除
            del(new_state_dict[key])
            # 再写入
            new_key_name = key.replace((key.split(".")[0] + "."), "")
            new_state_dict[new_key_name] = state_dict[key]

    return new_state_dict

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

# 计算psnr参数   
def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

# 光照信息解码函数，将光照信息恢复成RGB
def eval_sh(deg, sh, dirs):
    
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]
    
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]

    Returns:
        [..., C]
    """
    
    # 只能计算指定阶的函数
    assert deg <= 4 and deg >= 0
    # 看一下阶数是否和参数数量对应
    assert (deg + 1) ** 2 == sh.shape[-1]
    # 通道数，RGB为3通道
    C = sh.shape[-2]
    
    # 开始恢复RGB，计算先略过
    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])

    # [32768, 3]
    return result