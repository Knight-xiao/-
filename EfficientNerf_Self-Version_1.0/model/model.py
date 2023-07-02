import logging
import os
import time
from collections import defaultdict
from ctypes import sizeof
from pathlib import Path
from sys import getsizeof
from unittest import result

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from .net_block import Embedding, NeRF, NeRF_Rays
from .net_utils import eval_sh, get_rank, psnr, rewrite_weights_file_ckpt, rewrite_weights_file_rays_ckpt


class Efficient_NeRF(nn.Module):
    def __init__(self, sys_args):
        logging.info('创建Efficient Nerf模型中...')
        super(Efficient_NeRF, self).__init__()
        # 保存配置
        self.sys_args = sys_args
        # 获取数据集路径，为了获得类别数量
        self.data_root = self.sys_args['data']
        # 图片渲染尺寸
        self.img_wh = self.sys_args["default_res"]
        # 保存配置
        self.device = self.sys_args["device_type"]
        # 渲染的有效距离
        self.near = self.sys_args["near"]
        self.far = self.sys_args["far"]
        self.distance = self.far - self.near
        # 渲染区域的边界范围
        self.xyz_min = self.sys_args["global_boader_min"]
        self.xyz_max = self.sys_args["global_boader_max"]
        self.xyz_scope = self.xyz_max - self.xyz_min
        # chunk渲染数量
        self.chunk = self.sys_args["chunk"]
        # 粗采样数量，128
        self.coarse_numb = self.sys_args["coarse_numb"]
        # 粗采样空间网格数量
        self.grid_coarse = self.sys_args["grid_coarse"]
        # 初始化sigma查询空间使用的参数
        self.sigma_init = self.sys_args["sigma_init"]
        # 初始化
        self.sigma_default = self.sys_args["sigma_default"]
        # 初始化sigma全局调整的阶段轮数
        self.warmup_step = self.sys_args["warmup_step"]
        # 细采样的倍数
        self.sample_scale = self.sys_args["sample_fine_times"]
        # 网格空间划分倍数
        self.grid_scale = self.sys_args["grid_fine_times"]
        # 细采样数量， 128*5
        self.fine_numb = self.coarse_numb * self.sample_scale
        # 网格空间数量，384*3
        self.res_fine = self.grid_coarse * self.grid_scale
        # 渲染有效距离粗采样分份, 能这样直接near-far分块是因为计算渲染方向dir时，z=1
        self.z_vals_coarse = torch.linspace(self.near, self.far, self.coarse_numb, device=self.device)
        # 渲染有效距离细采样分份, 能这样直接near-far分块是因为计算渲染方向dir时，z=1
        self.z_vals_fine = torch.linspace(self.near, self.far, self.fine_numb, device=self.device)
        # 光线位置、方向编码器初始化，进NERF之前需要进行编码
        self.embedding_xyz = Embedding(self.sys_args["emb_freqs_xyz"]) # 10 is the default number
        self.embedding_cls = Embedding(self.sys_args["emb_freqs_cls"]) # 4 is the default number
        # 设定球谐函数的参数，阶数设定=deg+1，一般2-3阶够用了
        self.coarse_deg = self.sys_args["coarse_MLP_deg"]
        self.fine_deg = self.sys_args["fine_MLP_deg"]
        # 球谐函数在RGB三通道中应该有的基底系数数量，3阶是9个
        self.coarse_dim_sh = 3 * (self.coarse_deg + 1)**2
        self.fine_dim_sh = 3 * (self.fine_deg + 1)**2
        # 白色背景
        self.white_back = self.sys_args["white_back"]
        # 粗渲染阶段的sigam存储库学习速率
        self.coarse_beta = self.sys_args["coarse_beta"]
        # 粗采样生成权重的筛选阈值
        self.weight_thresh = self.sys_args['coarse_weight_thresh']
        # 训练过程中每轮训练完毕缓存图片路径
        self.train_img_pth = self.sys_args["demo_render_pth"]

        # 获取参数
        if self.sys_args["mode"] == 0:
            self.is_train = True
            self.is_distill = False
            self.batch = self.sys_args["batch"]
            # 全部的类别数量
            self.total_class_numb = len(os.listdir(self.data_root))
            # 保存索引时间
            self.extract_time = False
            # 调过粗渲染标志
            self.skip_coarse = False
            # 例化N个NERF模型
            for i in range(self.total_class_numb):
                self.add_module("coarse_nerf_{}".format(i), NeRF(self.sys_args, type="coarse").to(self.device))

            self.nerf_fine = NeRF(self.sys_args, type="fine")
            self.nerf_fine.train()
            self.nerf_rays = NeRF_Rays(self.sys_args)
            self.nerf_rays.train()

            # 体密度和体密度的索引信息, grid_coarse = 384 
            self.sigma_voxels_coarse = torch.full((self.total_class_numb, self.grid_coarse, self.grid_coarse, self.grid_coarse), self.sigma_init, device=self.device)
            self.idx_voxels_fine = torch.zeros((self.total_class_numb, self.grid_coarse, self.grid_coarse, self.grid_coarse), dtype=torch.bool, device=self.device)
            # 全局训练调整轮数，每渲染一次+1
            self.global_step = 0
            # 目前阶段细渲染光线的数量
            self.fine_rays_idx_numb = 0
        else:  
            self.is_train = False
            self.is_distill = False
            self.batch = self.sys_args["batch"]
            
            # 例化fine的ENERF模型，测试时候coarse不存在
            self.nerf_fine = NeRF(self.sys_args, type="fine")
            # self.nerf_rays = NeRF_Rays(self.sys_args)
            
            # 权重的保存路径
            self.nerf_ckpt_pth = self.sys_args['root_weight']
            # 权重文件名称
            self.nerf_ckpt_name = self.sys_args['nerf_model_name']
            # 重新修改一部分变量名称
            self.fine_nerf_ckpt = torch.load(os.path.join(Path(self.nerf_ckpt_pth), Path(self.nerf_ckpt_name)), map_location = self.device)
            self.idx_voxels_fine  = self.fine_nerf_ckpt["weight_voxels_fine"]
            
            self.fine_nerf_ckpt = rewrite_weights_file_ckpt(self.fine_nerf_ckpt)
            # self.rays_nerf_ckpt = rewrite_weights_file_rays_ckpt(self.fine_nerf_ckpt)     
            
            # 模型载入权重
            self.nerf_fine.load_state_dict(self.fine_nerf_ckpt)
            # self.nerf_rays.load_state_dict(self.rays_nerf_ckpt)
            
            self.global_step = 0

            logging.info("载入权重：{}".format(self.nerf_ckpt_name.split(".")[0]))

    # 前项计算过程
    def forward(self, rays, clss):
        self.global_step += 1
        # 渲染光线
        if self.is_train:
            render_result = self.render_rays(rays, clss)
        elif self.is_distill:
            render_result = self.distill_rays(rays, clss)
        else:
            render_result = self.render_rays_demo_fine(rays, clss)
            # render_result = self.render_rays_demo_rays(rays, clss)

        # 返回一个batch渲染的所有信息
        return render_result

    # [batch, cls_numb, 6]
    # [batch, cls_numb, 1]
    def render_rays(self, rays, clss):
        # 获取当前渲染光线的数目
        self.rays_numb = rays.shape[0]
        assert self.rays_numb == clss.shape[0], "rays and class are not matching!"

        # 渲染函数输出结果容器
        result = {}
        # 将rays拆解，之前处理的时候将rays_o和rays_d合并起来了
        # [batch, numb_cls, 3], [batch, numb_cls, 3]
        rays_o, rays_d = rays[..., 0:3], rays[..., 3:6] 
        # 将分份后的采样方法应用于每一条射线, [self.batch, self.total_class_numb, 128]
        z_vals_coarse_all = self.z_vals_coarse.clone().expand(self.rays_numb, self.total_class_numb, -1)
        # self.distance = 4，表示渲染的有效距离，far - near
        # 从(far-near)/128中取self.batch数量的数值，取数的规则是按均匀分布进行
        delta_z_vals_init = torch.empty(self.rays_numb, self.total_class_numb, 1, device=self.device).uniform_(0.0, self.distance/self.coarse_numb)
        # 训练阶段随机初始化每个采样间隔中间的采样点
        # 因为在体渲染算法中，每一小段间隔对应的sigma均相同，为了保证取到任意采样位置均能获得准确的sigma
        # 所以需要每次训练前随机在每个间隔段进行采样
        z_vals_coarse_all = z_vals_coarse_all + delta_z_vals_init

        # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
        # 注意，这里有相加关系，将相机位置与near-far关联起来
        # 按照原则，三维渲染物体标签的xyz位置信息应该在near和far之间
        # [self.batch, numb_cls, 128, 3]
        xyz_sampled_coarse = rays_o.unsqueeze(2) + rays_d.unsqueeze(2)*z_vals_coarse_all.unsqueeze(3)
 
        for cls_id in range(xyz_sampled_coarse.shape[1]):
            # 分类别计算
            xyz_sampled_coarse_single_cls = xyz_sampled_coarse[:, cls_id, ...]
            # 将128段粗采样分开，直接变成一小段一小段的光线信息        
            xyz_coarse = xyz_sampled_coarse_single_cls.reshape(-1, 3)

            # [self.batch, 1]
            cur_clss = clss[:,cls_id,:]
            cur_rays_d = rays_d[:,cls_id,:]
            cur_rays_o = rays_o[:,cls_id,:]

            if not self.skip_coarse: 
                # 集体查询，直接获取batch数量的sigma信息
                # 查找对应粗采样位置的各个sigma数值
                # [self.batch, 128]
                cur_cls_sigmas = self.query_coarse_sigma(xyz_coarse, cls_id).reshape(self.rays_numb, self.coarse_numb)
                
                # 只要不是在最后一轮的训练过程中
                with torch.no_grad():
                    cur_cls_sigmas[torch.rand_like(cur_cls_sigmas[:, 0]) < 0.01] = self.sigma_init
                    if self.warmup_step > 0 and self.global_step <= self.warmup_step:
                        # 训练的初始阶段，所有初始化的sigma均参与调整
                        # [self.batch*128, 2]
                        idx_render_coarse = torch.nonzero(cur_cls_sigmas >= -1e10).detach()
                    else:
                        # 过了全调整阶段，只调整有效的sigma
                        # [X, 2]
                        idx_render_coarse = torch.nonzero(cur_cls_sigmas > 0.0).detach()

                # cur_rgb_final_coarse: 粗采样生成的光线rgb颜色信息 [self.batch, 3]
                # cur_sigmas_coarse: 粗采样过程中每一段的权重 [self.batch, 128]
                # cur_rgbs_coarse: 粗采样过程中每一段的体密度 [self.batch, 128, 3]
                # print("############ coarse inference #############")
                cur_rgb_final_coarse, cur_sigmas_coarse, cur_weights_coarse, cur_rgbs_coarse = \
                self.inference(self._modules["coarse_nerf_{}".format(cls_id)], 
                               self.embedding_xyz, 
                               self.embedding_cls, 
                               xyz_sampled_coarse_single_cls, 
                               cur_rays_d,
                               cur_clss, 
                               z_vals_coarse_all[:,cls_id,:], 
                               idx_render_coarse,
                               mode = "coarse")
            
                if cls_id == 0:
                    cat_rgb_coarse = cur_rgb_final_coarse.unsqueeze(1)
                else:
                    cat_rgb_coarse = torch.cat([cat_rgb_coarse, cur_rgb_final_coarse.unsqueeze(1)], dim=1)

                # 筛选出有效的渲染光线
                xyz_coarse_idx = xyz_sampled_coarse_single_cls[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
                # 筛选出有效的sigma体密度
                sigmas_coarse_idx = cur_sigmas_coarse.detach()[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
                # 更新coarse阶段的sigma查询表
                self.update_coarse_sigma(xyz_coarse_idx, sigmas_coarse_idx, self.coarse_beta, cls_id)
            
                # 在更新后的sigma库里再次走一遍查询计算
                # 拿到新的权重参与后续的fine部分调整与更新
                with torch.no_grad():
                    # 将光路上的delta分为128份，最后一份无限长
                    # [self.batch, 127]
                    deltas_coarse = z_vals_coarse_all[:,cls_id,1:] - z_vals_coarse_all[:, cls_id, :-1]
                    delta_inf = 1e10 * torch.ones_like(deltas_coarse[:, :1]) 
                    # [self.batch, 128]
                    deltas_coarse = torch.cat([deltas_coarse, delta_inf], -1)
                    # 更新后的weights信息
                    cur_cls_weights_coarse = self.sigma2weights(deltas_coarse, cur_cls_sigmas)
                    # [self.batch, 128]
                    cur_cls_weights_coarse = cur_cls_weights_coarse.detach()

                idx_render = torch.nonzero(cur_cls_weights_coarse >= min(self.weight_thresh, cur_cls_weights_coarse.max().item()))                
                idx_render = idx_render.unsqueeze(1).expand(-1, self.sample_scale, -1)
                idx_render_fine = idx_render.clone()
                idx_render_fine[..., 1] = idx_render[..., 1] * self.sample_scale + (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale)
                idx_render_fine = idx_render_fine.reshape(-1, 2)

                z_vals_fine_all = self.z_vals_fine.clone().expand(self.rays_numb, -1)
                # 训练阶段随机初始化每个采样间隔中间的采样点
                z_vals_fine_all = z_vals_fine_all + delta_z_vals_init[:,cls_id,:]
                xyz_sampled_fine = cur_rays_o.unsqueeze(1) + cur_rays_d.unsqueeze(1) * z_vals_fine_all.unsqueeze(2)   

            # 查找渲染
            else:
                with torch.no_grad():
                    z_vals_fine_all = self.z_vals_fine.clone().expand(self.rays_numb, -1)
                    z_vals_fine_all = z_vals_fine_all + delta_z_vals_init[:,cls_id,:]
                    # [batch, 640, 3]
                    xyz_sampled_fine = cur_rays_o.unsqueeze(1) + cur_rays_d.unsqueeze(1) * z_vals_fine_all.unsqueeze(2)  
                    # [batch*640, 3]
                    xyz_fine = xyz_sampled_fine.reshape(-1, 3)
                    # [batch, 640]，内容是True/False的Flag
                    idx_render_fine = self.query_fine_index(xyz_fine, cls_id).reshape(self.rays_numb, -1)
                    idx_render_fine = torch.nonzero(idx_render_fine)

            # batch数量限制
            if idx_render_fine.shape[0] > self.batch * 128:
                indices = torch.randperm(idx_render_fine.shape[0])[:self.batch * 128]
                idx_render_fine = idx_render_fine[indices]

            # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
            # 注意，这里有相加关系，将相机位置与near-far关联起来
            # 按照原则，三维渲染物体标签的xyz位置信息应该在near和far之间
            rgb_fine, sigmas_fine, weights_fine, rgbs_fine = \
            self.inference(self.nerf_fine, 
                            self.embedding_xyz, 
                            self.embedding_cls, 
                            xyz_sampled_fine, 
                            cur_rays_d,
                            cur_clss,  
                            z_vals_fine_all, 
                            idx_render_fine,
                            mode = "fine")

            if (not self.skip_coarse) and (self.extract_time):
                # numb_obj_sample = 16
                # # weights_fine: [batch, sample]
                # # xyz_sampled_fine: [batch, sample, 3]
                
                # # 去掉第一个权重和最后一个权重，其余权重全部应该是object的权重
                # # [batch, sample-2]
                # weights_fine_obj = weights_fine[:, 1:-1]
                # # 将权重从大到小排列，默认按行 [batch, sample-2]
                # # [batch, sample-2]
                # weights_fine_obj, idx_w_obj = torch.sort(weights_fine_obj, descending=True)
                # # 取多少个点表示obj
                # # [N, 2]
                # idx_contribute_obj = torch.nonzero(weights_fine_obj[:, :numb_obj_sample] >=self.weight_thresh)
                # # [N, 16, 2]
                # idx_w_obj = torch.nonzero(idx_w_obj[:, :numb_obj_sample] >= 0).reshape(weights_fine_obj.shape[0], numb_obj_sample, 2)
                # # 所有坐标的纵坐标需要+1,因为跳过了起始点和终止点
                # idx_w_obj[..., 1] = idx_w_obj[..., 1] + 1
                # # 起始和终止点加上
                # start_idx = idx_w_obj[:, 0].clone().unsqueeze(1) 
                # end_idx = idx_w_obj[:, 0].clone().unsqueeze(1)
                # start_idx[..., 1] = 0
                # end_idx[..., 1] = weights_fine.shape[-1]-1
                # # 合并
                # idx_w_obj = torch.cat([start_idx, idx_w_obj, end_idx], dim=1)

                # # 考虑完大小、有效度之后的索引
                # final_idx_render = idx_w_obj[idx_contribute_obj[:,0], idx_contribute_obj[:,1]]

                # xyz_fine_weights_choice = xyz_sampled_fine[final_idx_render[:,0], final_idx_render[:,1]]
                # self.update_fine_index(xyz_fine_weights_choice, cls_id)

                final_idx_render = torch.nonzero(weights_fine >= min(self.weight_thresh, weights_fine.max().item()))
                mask_for_start = final_idx_render[:, 1] < 200
                mask_for_end   = final_idx_render[:, 1] > 600

                mask_start_with_end = mask_for_start + mask_for_end

                final_idx_render = final_idx_render[~mask_start_with_end]

                xyz_fine_weights_choice = xyz_sampled_fine[final_idx_render[:,0], final_idx_render[:,1]]
                self.update_fine_index(xyz_fine_weights_choice, cls_id)
                
            if cls_id == 0:
                cat_rgb_fine = rgb_fine.unsqueeze(1)
            else:
                cat_rgb_fine = torch.cat([cat_rgb_fine, rgb_fine.unsqueeze(1)], dim=1)
                
        if self.skip_coarse:
            result['rgb_fine'] = cat_rgb_fine
        else:
            result['rgb_coarse'] = cat_rgb_coarse
            result['rgb_fine'] = cat_rgb_fine

        return result
    
    # [batch, cls_numb, 6]
    # [batch, cls_numb, 1]
    def distill_rays(self, rays, clss):
        # 获取当前渲染光线的数目
        self.rays_numb = rays.shape[0]
        assert self.rays_numb == clss.shape[0], "rays and class are not matching!"

        # 渲染函数输出结果容器
        result = {}
        # [batch, numb_cls, 3], [batch, numb_cls, 3]
        rays_o, rays_d = rays[..., 0:3], rays[..., 3:6] 
        # [self.batch, self.total_class_numb, 640]
        z_vals_fine_all = self.z_vals_fine.clone().expand(self.rays_numb, self.total_class_numb, -1)
        delta_z_vals_init = torch.empty(self.rays_numb, self.total_class_numb, 1, device=self.device).uniform_(0.0, self.distance/self.coarse_numb)
        z_vals_fine_all = z_vals_fine_all + delta_z_vals_init
        # [self.batch, numb_cls, 640, 3]
        xyz_sampled_fine = rays_o.unsqueeze(2) + rays_d.unsqueeze(2)*z_vals_fine_all.unsqueeze(3)
 
        for cls_id in range(xyz_sampled_fine.shape[1]):
            # [self.batch, 640, 3]
            xyz_sampled_fine_single_cls = xyz_sampled_fine[:, cls_id, ...]
            # [self.batch*640, 3]      
            xyz_fine = xyz_sampled_fine_single_cls.reshape(-1, 3)
            # [self.batch, 1]
            cur_clss = clss[:,cls_id,:]
            # [self.batch, 3]
            cur_rays_d = rays_d[:,cls_id,:]

            # 查找渲染计算前项
            with torch.no_grad():
                # [batch, 640]，内容是True/False的Flag
                idx_render_fine = self.query_fine_index(xyz_fine, cls_id).reshape(self.rays_numb, -1)
                idx_render_fine = torch.nonzero(idx_render_fine)

                # batch数量限制
                if idx_render_fine.shape[0] > self.batch * 128:
                    indices = torch.randperm(idx_render_fine.shape[0])[:self.batch * 128]
                    idx_render_fine = idx_render_fine[indices]

                rgb_fine, sigmas_fine, weights_fine, rgbs_fine= \
                self.inference(self.nerf_fine, 
                                self.embedding_xyz, 
                                self.embedding_cls, 
                                xyz_sampled_fine_single_cls, 
                                cur_rays_d,
                                cur_clss,  
                                z_vals_fine_all[:,cls_id,:], 
                                idx_render_fine,
                                mode = "fine")

            ray_rgbs = self.inference_rays(self.nerf_rays,
                                            self.embedding_xyz, 
                                            self.embedding_cls, 
                                            xyz_sampled_fine_single_cls, 
                                            cur_clss, 
                                            idx_render_fine)
                
            if cls_id == 0:
                cat_rgb_rays = ray_rgbs.unsqueeze(1)
                cat_rgb_fine = rgb_fine.unsqueeze(1)
            else:
                cat_rgb_rays = torch.cat([cat_rgb_rays, ray_rgbs.unsqueeze(1)], dim=1)
                cat_rgb_fine = torch.cat([cat_rgb_fine, rgb_fine.unsqueeze(1)], dim=1)

        result['rgb_rays'] = cat_rgb_rays
        result['rgb_fine'] = cat_rgb_fine

        return result        

    # 输入rays和clss经过排序，从小到大按类别
    # [batch, 6]
    # [batch, 1]
    # 输入640细光线查询结果
    def render_rays_demo_fine(self, rays, clss):
        self.rays_numb = rays.shape[0]
        clss_type, clss_numb = torch.unique(clss, return_counts=True)
        
        fine_numb = 512
        z_vals_fine = torch.linspace(self.near, self.far, fine_numb, device=self.device)
        
        for idx, cur_clss in enumerate(clss_type):
            cut_rays = rays[:clss_numb[idx], :]
            cur_clss = clss[:clss_numb[idx], :]
            cur_rays_numb = cut_rays.shape[0]
            # cur_clss
            rays = rays[clss_numb[idx]:, :]
            # 渲染函数输出结果容器
            result = []        
            # 将rays拆解，之前处理的时候将rays_o和rays_d合并起来了
            # [self.batch, 3], [self.batch, 3]
            cur_rays_o, cur_rays_d = cut_rays[:, 0:3], cut_rays[:, 3:6]
            # 将分份后的采样方法应用于每一条射线, [self.batch, 128]
            z_vals_fine_all = z_vals_fine.clone().expand(cur_rays_numb, -1)

            # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
            # 注意，这里有相加关系，将相机位置与near-far关联起来
            # 按照原则，三维渲染物体标签的xyz位置信息应该在near和far之间
            # [self.batch, 128, 3]
            xyz_sampled_fine = cur_rays_o.unsqueeze(1) + cur_rays_d.unsqueeze(1)*z_vals_fine_all.unsqueeze(2)
            # [self.batch*128, 3]
            xyz_fine_for_index = xyz_sampled_fine.reshape(-1, 3)
           
            idx_render_flag = self.query_fine_index(xyz_fine_for_index, idx).reshape(cur_rays_numb, fine_numb)
            # 1/10目前大约
            idx_render_fine = torch.nonzero(idx_render_flag)

            # out_chunks = []
            if idx_render_fine.shape[0] > self.batch * 16:
                indices = torch.randperm(idx_render_fine.shape[0])[:self.batch * 64]
                idx_render_fine = idx_render_fine[indices]
            # 当查不到光线占据格的时候直接填充RGB
            if idx_render_fine.shape[0] == 0:
                rgb_final = torch.ones([xyz_sampled_fine.shape[0], 3], device=self.device)
            else:
                rgb_final, sigmas_fine, weights_fine, rgbs_fine = \
                self.inference(self.nerf_fine, 
                            self.embedding_xyz, 
                            self.embedding_cls, 
                            xyz_sampled_fine, 
                            cur_rays_d,
                            cur_clss,  
                            z_vals_fine_all, 
                            idx_render_fine,
                            mode = fine_numb)            
            
            result += [rgb_final]
        
        result = torch.cat(result, 0)
        
        return result

    def render_rays_demo_rays(self, rays, clss):
        self.rays_numb = rays.shape[0]
        clss_type, clss_numb = torch.unique(clss, return_counts=True)
        
        fine_numb = 640
        z_vals_fine = torch.linspace(self.near, self.far, fine_numb, device=self.device)
        
        for idx, cur_clss in enumerate(clss_type):
            cut_rays = rays[:clss_numb[idx], :]
            cur_clss = clss[:clss_numb[idx], :]
            cur_rays_numb = cut_rays.shape[0]
            # cur_clss
            rays = rays[clss_numb[idx]:, :]
            # 渲染函数输出结果容器
            result = []        
            # 将rays拆解，之前处理的时候将rays_o和rays_d合并起来了
            # [self.batch, 3], [self.batch, 3]
            cur_rays_o, cur_rays_d = cut_rays[:, 0:3], cut_rays[:, 3:6]
            # 将分份后的采样方法应用于每一条射线, [self.batch, 128]
            z_vals_fine_all = z_vals_fine.clone().expand(cur_rays_numb, -1)

            # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
            # 注意，这里有相加关系，将相机位置与near-far关联起来
            # 按照原则，三维渲染物体标签的xyz位置信息应该在near和far之间
            # [self.batch, 128, 3]
            xyz_sampled_fine = cur_rays_o.unsqueeze(1) + cur_rays_d.unsqueeze(1)*z_vals_fine_all.unsqueeze(2)

            rgb_rays = self.inference_rays(self.nerf_rays, 
                                            self.embedding_cls, 
                                            xyz_sampled_fine, 
                                            cur_clss)         
            
            result += [rgb_rays]
        
        result = torch.cat(result, 0)
        
        return result
    
    # 输入128粗光线，查询结果
    def render_rays_demo_coarse(self, rays, clss):
        clss_type, clss_numb = torch.unique(clss, return_counts=True)
        for idx, cur_clss in enumerate(clss_type):
            cut_rays = rays[:clss_numb[idx], :]
            cur_clss = clss[:clss_numb[idx], :]
            cur_rays_numb = cut_rays.shape[0]
            # cur_clss
            rays = rays[clss_numb[idx]:, :]
            # 渲染函数输出结果容器
            result = []        
            # 将rays拆解，之前处理的时候将rays_o和rays_d合并起来了
            # [self.batch, 3], [self.batch, 3]
            cur_rays_o, cur_rays_d = cut_rays[:, 0:3], cut_rays[:, 3:6]
            # 将分份后的采样方法应用于每一条射线, [self.batch, 128]
            z_vals_coarse_all = self.z_vals_coarse.clone().expand(cur_rays_numb, -1)
            # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
            # 注意，这里有相加关系，将相机位置与near-far关联起来
            # 按照原则，三维渲染物体标签的xyz位置信息应该在near和far之间
            # [self.batch, 128, 3]
            xyz_sampled_coarse = cur_rays_o.unsqueeze(1) + cur_rays_d.unsqueeze(1)*z_vals_coarse_all.unsqueeze(2)
            # [self.batch*128, 3]
            xyz_coarse = xyz_sampled_coarse.reshape(-1, 3)
           
            idx_render_flag = self.query_fine_index(xyz_coarse, idx).reshape(cur_rays_numb, self.coarse_numb)
            idx_render_fine = torch.nonzero(idx_render_flag)
          
            xyz_coarse = xyz_sampled_coarse[idx_render_fine[:, 0], idx_render_fine[:, 1]].reshape(-1, 3)
            dim_sh = self.fine_dim_sh

            view_dir = cur_rays_d.unsqueeze(1).expand(-1, self.coarse_numb, -1)
            ray_clss = cur_clss.unsqueeze(1).expand(-1, self.coarse_numb, -1)

            view_dir = view_dir[idx_render_fine[:, 0], idx_render_fine[:, 1]] 
            ray_clss = ray_clss[idx_render_fine[:, 0], idx_render_fine[:, 1]]                     
            # 记录筛选后还剩信息的数量
            cur_numb_part = xyz_coarse.shape[0]
            # 输出信息的缓存容器
            out_chunks = []
            
            # 对batch进行了合并, 将batch分成了128份，而渲染函数只能每次渲染一段，所以还需要循环
            for i in range(0, cur_numb_part, self.chunk):
                # 对输入信息编码，注意，方向信息没有编码
                input_encode_xyz = self.embedding_xyz(xyz_coarse[i:i+self.chunk])
                input_encode_cls = self.embedding_cls(ray_clss[i:i+self.chunk])
                input_dir = view_dir[i:i+self.chunk]
                # 计算输出信息，输出信息中有:
                # out = torch.cat([sigma, rgb, sh], -1)
                nerf_model_out = self.nerf_fine(input_encode_xyz, input_encode_cls, input_dir)
                out_chunks += [nerf_model_out]
            
            out_chunks = torch.cat(out_chunks, 0)

            # 填充数据，其他未参与细渲染计算的结果全部用默认值
            out_rgb = torch.full((cur_rays_numb, self.coarse_numb, 3), 1.0, device=self.device)
            out_sigma = torch.full((cur_rays_numb, self.coarse_numb, 1), self.sigma_default, device=self.device)
            out_sh = torch.full((cur_rays_numb, self.coarse_numb, dim_sh), 0.0, device=self.device)

            out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)
            out_defaults[idx_render_fine[:, 0], idx_render_fine[:, 1]] = out_chunks                     
                
            # 拆分信息
            # sigmas: [self.batch, 128, 1]
            # rgbs_pred: [self.batch, 128, 3]
            # shs: [self.batch, 128, 27]
            sigmas_pred, rgbs_fine, _ = torch.split(out_defaults, (1, 3, dim_sh), dim=-1) 
            # 修正格式变为[self.batch, fine_numb]
            sigmas_fine = sigmas_pred.squeeze(-1)

            # ##############
            # 6.获取fine_delta
            # ##############
            deltas_fine = z_vals_coarse_all[:, 1:] - z_vals_coarse_all[:, :-1]
            delta_inf_fine = 1e10 * torch.ones_like(deltas_fine[:, :1])
            # [self.batch, fine_numb] 
            deltas_fine = torch.cat([deltas_fine, delta_inf_fine], -1) 

            # ##############
            # 7.sigma,delta查询weights_fine
            # ##############
            weights_fine = self.sigma2weights(deltas_fine, sigmas_fine)
            weights_sum = weights_fine.sum(1) 

            # ##############
            # 8. 计算最终渲染颜色
            # ##############
            rgb_final = torch.sum(weights_fine.unsqueeze(-1)*rgbs_fine, -2)

            if self.white_back:
                rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

            result += [rgb_final]
        
        result = torch.cat(result, 0)
        
        return result

    # 计算光线输入模型的具体过程
    # model:例化模型
    # embedding_xyz: 位置编码函数
    # embedding_dir: 方向编码函数
    # xyz:[self.batch, 128, 3]
    # dir:[self.batch, 3]
    # z_vals: 每条光线的分份采样 [self.batch, 128]
    # idx_render: 从查询库中取出的查询信息位置索引编号 [self.batch*128, 2]/[X, 2]
    def inference(self, model, embedding_xyz, embedding_cls, xyz, dir, clss, z_vals, idx_render, mode):        
        if mode == "coarse":
            sample_numb = self.coarse_numb
            dim_sh = self.coarse_dim_sh
        elif mode == "fine":
            sample_numb = self.fine_numb
            dim_sh = self.fine_dim_sh
        else:
            sample_numb = mode
            dim_sh = self.fine_dim_sh
        # 选出渲染光路的128份中对渲染最终结果有效果sigma所在的段落
        # xyz:[self.batch*128, 3]
        # 经过筛选，在warmup之前依然是[self.batch*128, 3]
        # warmup经过之后变成[X, 3]
        xyz = xyz[idx_render[:, 0], idx_render[:, 1]].view(-1, 3)
        
        # 经过筛选，在warmup之前依然是[self.batch*128, 3]
        # warmup经过之后变成[X, 3]
        # 找到哪些入射角度的光线是对渲染有效的
        view_dir = dir.unsqueeze(1).expand(-1, sample_numb, -1)
        ray_clss = clss.unsqueeze(1).expand(-1, sample_numb, -1)

        view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]
        ray_clss = ray_clss[idx_render[:, 0], idx_render[:, 1]]        
        # 记录筛选后还剩信息的数量
        cur_numb_part = xyz.shape[0]

        if mode == "fine":
            self.fine_rays_idx_numb = cur_numb_part
        
        # 输出信息的缓存容器
        out_chunks = []
        model.train()
        # 对batch进行了合并, 将batch分成了128份，而渲染函数只能每次渲染一段，所以还需要循环
        for i in range(0, cur_numb_part, self.chunk):
            # 对输入信息编码，注意，方向信息没有编码
            input_encode_xyz = embedding_xyz(xyz[i:i+self.chunk])
            input_encode_cls = embedding_cls(ray_clss[i:i+self.chunk])
            input_dir = view_dir[i:i+self.chunk]
            # 计算输出信息，输出信息中有:
            nerf_model_out = model(input_encode_xyz, input_encode_cls, input_dir)
            
            if i == 0:
                out_chunks = nerf_model_out
            else:
                out_chunks = torch.cat([out_chunks, nerf_model_out], 0)

        # 初始化容器，承接输出信息
        out_rgb = torch.full((self.rays_numb, sample_numb, 3), 1.0, device=self.device)
        out_sigma = torch.full((self.rays_numb, sample_numb, 1), self.sigma_default, device=self.device)
        out_sh = torch.full((self.rays_numb, sample_numb, dim_sh), 0.0, device=self.device)
        out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)

        out_defaults[idx_render[:, 0], idx_render[:, 1]] = out_chunks
        
        # 拆分信息
        # sigmas: [self.batch, 128, 1]
        # rgbs: [self.batch, 128, 3]
        # shs: [self.batch, 128, 27]
        sigmas, rgbs, shs = torch.split(out_defaults, (1, 3, dim_sh), dim=-1)    
        
        del out_chunks
        del out_defaults

        # sigmas: [self.batch, 128]
        sigmas = sigmas.squeeze(-1)
        # print(sigmas, "nnnnn")

        # 拆分出每一个采样小段的delta采样数值，分成128段，采样127个
        # [self.batch, 127]
        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        # 第128段则表示无限长度的采样信息，如果当前光线经过物体，则该项几乎没用，因为权重为0
        # 如果当前光线不经过物体，则该项直接影响颜色
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) 
        # 合并成128项, [self.batch, 128]
        deltas = torch.cat([deltas, delta_inf], -1)        
        # print(deltas, "dddddd")
        # 通过deltas和sigmas计算出相应小采样段落对最终颜色的贡献权重
        # [self.batch, 128]
        weights = self.sigma2weights(deltas, sigmas)
        # print(weights.shape, "wwwwwww")
        # [self.batch]
        weights_sum = weights.sum(1) 
        # print(weights_sum, "wswswswswswsws")
        # 每一段rgb颜色与相应的权重相乘，再相加，就计算除了这一条光线应该呈现的总颜色
        # [self.batch, 3]
        rgbs_weights = weights.unsqueeze(-1)*rgbs
        rgb_final = torch.sum(rgbs_weights, -2)

        if self.white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        # rgb_final: [self.batch, 3]
        # weights: [self.batch, 128]
        # sigmas: [self.batch, 128]
        # shs: [self.batch, 128, 27]
        return rgb_final, sigmas, weights, rgbs_weights
        # return rgb_final, weights, sigmas, shs

    # xyz:[self.batch, 128, 3]
    # clss:[self.batch, 1]
    def inference_rays(self, model, embedding_xyz, embedding_cls, xyz, clss, idx_render=None):
        # [N, 63]
        xyz = xyz.reshape(xyz.shape[0], -1)
        input_encode_xyz = embedding_xyz(xyz)
        input_encode_cls = embedding_cls(clss)
        
        # [Batch, 3]
        ray_rgb = model(input_encode_xyz, input_encode_cls)

        # shs: [self.batch, 128, 27]
        return ray_rgb
    
    # xyz:[self.batch, 128, 3]
    # clss:[self.batch, 1]
    def inference_rays_bk(self, model, embedding_xyz, embedding_cls, xyz, clss, idx_render=None):
      
        xyz = xyz.reshape(xyz.shape[0], -1)
        input_encode_xyz = embedding_xyz(xyz)
        input_encode_cls = embedding_cls(clss)
        # print(input_encode_cls.shape)

        # [Batch, 3]
        ray_rgb = model(input_encode_xyz, input_encode_cls)

        # shs: [self.batch, 128, 27]
        return ray_rgb
    # 在固定的索引空间中查找相应的透明度信息
    # 如果是测试，就直接读取权重文件
    # 如果是训练，就直接在优化中的索引空间中获取，最后把索引空间存下来
    def query_coarse_sigma(self, xyz, idx_cls):
        # 将当前渲染的光线信息嵌入到全局场景的范围内
        # [13107200, 3])
        ijk_coarse = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse-1)
        # print(self.xyz_min, self.xyz_scope, self.grid_coarse, "#2323")
        # 查询相应的不透明度
        sigmas = self.sigma_voxels_coarse[idx_cls, ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
        
        return sigmas

    # 输入xyz:[batch, 128, 3]
    # 表示一条光线分成coarse_sample份进行每一段的查询
    def query_fine_index(self, xyz, idx_cls):
        # 将当前渲染的光线信息嵌入到全局场景的范围内
        ijk_fine = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse-1)
        # index_voxels_coarse中保存了有效sigma的位置
        # [384, 384, 384]
        # test_voxels = torch.zeros_like(self.idx_voxels_fine, dtype=torch.bool)
        # test_voxels[idx_cls, 150:250, 150:250, 150:250] = self.idx_voxels_fine[idx_cls, 150:250, 150:250, 150:250]
        # print("33333333")
        # idx = test_voxels[idx_cls, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]]
        idx = self.idx_voxels_fine[idx_cls, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]]
        
        return idx

    # 利用sigma和delta计算颜色生成过程中的权重信息
    # 对应论文中的公式(2)中的wi
    def sigma2weights(self, deltas, sigmas):
        # 初始化噪声，噪声分布0-1的正态分布，尺寸与不透明度相同
        # 增强网络的适应性
        noise = torch.randn(sigmas.shape, device=self.device)
        sigmas = sigmas + noise

        # [self.batch, 128]
        alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas))
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        
        # torch.cumprod与torch.cumsum类似，按维度累乘
        # [self.batch, 128]
        # 直接计算得到了公式(2)中的wi
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] 

        # [self.batch, 128]
        return weights

    # 更新粗渲染过程的体密度信息
    def update_coarse_sigma(self, xyz, sigma, beta, idx_cls):
            # 将当前渲染的光线信息嵌入到全局场景的范围内
        ijk_coarse = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse-1)
        # 更新信息
        self.sigma_voxels_coarse[idx_cls, ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] \
                   = (1 - beta)*self.sigma_voxels_coarse[idx_cls, ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] + beta*sigma

    # 更新权重
    def update_fine_index(self, xyz, idx_cls):
        # 获取粗渲染存储容器下粗渲染有效信息的索引
        ijk_fine = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse-1)
        self.idx_voxels_fine[idx_cls, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]] = True

    def rewrite_ckpt_before_distill(self, ckpt_file):
        logging.info("载入目标蒸馏模型：{}".format(ckpt_file.split("/")[-1]))
        # 获取权重信息
        ckpt_file = torch.load(ckpt_file, map_location = self.device)
        self.idx_voxels_fine = ckpt_file["weight_voxels_fine"]
        state_dict = ckpt_file["model"]
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

    # 验证细渲染的训练结果
    def valid_fine(self, valid_rays_loader, epoch):
        # 首先清空缓存
        torch.cuda.empty_cache()
        rank = get_rank()
            
        # 分布式情况下用0线程卡测试
        # 单卡模式默认rank为0
        if rank == 0:
            val_model = NeRF(self.sys_args, type="fine").to(self.device)
            val_model.eval()
            # 载入验证模型，生成验证结果
            logging.info("载入验证模型：{}".format(self.model_name))
            fine_ckpt = torch.load(self.file_path, map_location = self.device)["model"]
            
            # 编辑权重信息
            new_fine_ckpt = fine_ckpt.copy()
            for key in fine_ckpt:
                # 删除粗查询的权重
                if key.split(".")[0] != "nerf_fine":
                    del(new_fine_ckpt[key])
                # 重命名细粒度查询的相关信息
                else:
                    # 先删除
                    del(new_fine_ckpt[key])
                    # 再写入
                    new_key_name = key.replace((key.split(".")[0] + "."), "")
                    new_fine_ckpt[new_key_name] = fine_ckpt[key]
            
            # 载入权重
            val_model.load_state_dict(new_fine_ckpt)
            val_model.eval()
            # 缓存结果
            results_cat = {}
            rgb_cat = {}
            # 单张图片需要的所有光线数量
            single_img_rays_numb = self.img_wh * self.img_wh
            numb_loader = round(single_img_rays_numb / self.batch + 0.5)

            # 前项测试
            with torch.no_grad():
                logging.info("渲染图片中...")

                for idx, data in enumerate(valid_rays_loader):
                    if idx > numb_loader:
                        break
                    else:
                        # 获取渲染光线，data:[Batch, W*H, 6]
                        rgbs_data, rays_data, clss_data = data
                        # GPU
                        rays_data = rays_data.to(self.device) #[Batch, numb_clss, 3]
                        rgbs_data = rgbs_data.to(self.device) #[Batch, numb_clss, 3]
                        clss_data = clss_data.to(self.device) #[Batch, numb_clss, 1]
                        # 总种类数量
                        total_clss_numb = rays_data.shape[1]
                        # 循环处理每个类别
                        for cur_clss in range(total_clss_numb):
                            if idx == 0:
                                results_cat[cur_clss] = []
                                rgb_cat[cur_clss]  = []                 
                            # ##############
                            # 1.获取sigma
                            # ##############
                            rays_o, rays_d = rays_data[:, cur_clss, 0:3], rays_data[:, cur_clss, 3:6] # both (self.batch, 3)
                            clss = clss_data[:,cur_clss,:]
                            if not self.skip_coarse:
                                # 将分份后的采样方法应用于每一条射线, [self.batch, 128]
                                z_vals_coarse_all = self.z_vals_coarse.clone().expand(self.batch, -1)
                                # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
                                # [self.batch, 128, 3]
                                xyz_sampled_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1)*z_vals_coarse_all.unsqueeze(2)   
                                xyz_coarse = xyz_sampled_coarse.reshape(-1, 3)
                                # 集体查询，直接获取batch数量的sigma信息
                                # 查找对应粗采样位置的各个sigma数值
                                # [self.batch, 128]
                                sigmas_coarse = self.query_coarse_sigma(xyz_coarse, cur_clss).reshape(self.batch, self.coarse_numb)
                                # ##############
                                # 2.获取coarse_delta
                                # ##############
                                # 将光路上的delta分为128份，最后一份无限长
                                # [self.batch, 127]
                                deltas_coarse = z_vals_coarse_all[:, 1:] - z_vals_coarse_all[:, :-1]
                                delta_inf_coarse = 1e10 * torch.ones_like(deltas_coarse[:, :1]) 
                                # [self.batch, 128]
                                deltas_coarse = torch.cat([deltas_coarse, delta_inf_coarse], -1)   
                                # ##############
                                # 3.sigma,delta查询weights
                                # ##############
                                # [self.batch, 128]
                                weights_coarse = self.sigma2weights(deltas_coarse, sigmas_coarse)
                                # ##############
                                # 4.weights_coarse生成细渲染索引
                                # ##############
                                idx_render = torch.nonzero(weights_coarse >= min(self.weight_thresh, weights_coarse.max().item()))
      
                                # z_vals_fine数值为粗采样的sample_scale倍, 目前是128*5
                                # [self.batch, fine_numb]
                                z_vals_fine_all = self.z_vals_fine.clone().expand(self.batch, -1)
                                # [X, self.sample_scale, 2]
                                idx_render = idx_render.unsqueeze(1).expand(-1, self.sample_scale, -1)
                                idx_render_fine = idx_render.clone()
                            
                                # 索引信息转换，由粗渲染生成的索引转化为细渲染阶段索引
                                # idx_render格式中的最后一项表示第idx_render[..., 0]条光线的第idx_render[..., 1]段
                                # idx_render[..., 1]的范围是0~128，乘以sample_scale表示扩大5倍，将128采样变为128*5倍，索引间距也乘以了5倍
                                # idx_render[..., 1] * self.sample_scale 尺寸为 [X, 5]
                                # (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale) 尺寸为 [1, 5]
                                # 两者相加结果是[X, 5]
                                # 两者相加的含义是将复制了5份的索引间距更加细化，索引间距乘以5倍后中间的间距再细化为5份，复制的每一份分别对应细化的一个分度
                                idx_render_fine[..., 1] = idx_render[..., 1] * self.sample_scale + (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale)
                                # 再重新整合为索引格式，变为[X*sample_scale, 2]
                                idx_render_fine = idx_render_fine.reshape(-1, 2)
                                # 筛选有效光线
                                # [self.batch, 128*sample_scale, 3]
                                xyz_sampled_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine_all.unsqueeze(2) 
                            else:
                                z_vals_fine_all = self.z_vals_fine.clone().expand(self.batch, -1)
                                xyz_sampled_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine_all.unsqueeze(2)
                                xyz_fine_for_query = xyz_sampled_fine.reshape(-1, 3) 
                                idx_render_fine = self.query_fine_index(xyz_fine_for_query, cur_clss).reshape(self.batch, -1)
                                idx_render_fine = torch.nonzero(idx_render_fine)
                            
                            # ##############
                            # 5.计算fine_sigma, fine_rgbs
                            # ##############
                            dim_sh = self.fine_dim_sh
                            xyz_fine = xyz_sampled_fine[idx_render_fine[:, 0], idx_render_fine[:, 1]].reshape(-1, 3)
                            
                            if xyz_fine.shape[0] == 0:
                                # rgb_final = torch.zeros([xyz_sampled_fine.shape[0], 3], device=self.device)
                                rgb_final = torch.ones([xyz_sampled_fine.shape[0], 3], device=self.device)
                                # print(rgb_final.shape, "2222")
                            else:
                                # 筛选有效角度
                                # [self.batch, 128*sample_scale, 3]
                                view_dir = rays_d.unsqueeze(1).expand(-1, self.fine_numb, -1)
                                ray_clss = clss.unsqueeze(1).expand(-1, self.fine_numb, -1)

                                view_dir = view_dir[idx_render_fine[:, 0], idx_render_fine[:, 1]] 
                                ray_clss = ray_clss[idx_render_fine[:, 0], idx_render_fine[:, 1]]                     
                                # 记录筛选后还剩信息的数量
                                cur_numb_part = xyz_fine.shape[0]
                                # 输出信息的缓存容器
                                out_chunks = []
                                # 对batch进行了合并, 将batch分成了128份，而渲染函数只能每次渲染一段，所以还需要循环
                                for i in range(0, cur_numb_part, self.chunk):
                                    # print(xyz_fine[i:i+self.chunk].shape, "222")
                                    # 对输入信息编码，注意，方向信息没有编码
                                    input_encode_xyz = self.embedding_xyz(xyz_fine[i:i+self.chunk])
                                    input_encode_cls = self.embedding_cls(ray_clss[i:i+self.chunk])
                                    # print(input_encode_xyz.shape, "666")
                                    # print(input_encode_cls.shape, "777")
                                    input_dir = view_dir[i:i+self.chunk]
                                    # 计算输出信息，输出信息中有:
                                    # out = torch.cat([sigma, rgb, sh], -1)
                                    nerf_model_out = val_model(input_encode_xyz, input_encode_cls, input_dir)
                                    out_chunks += [nerf_model_out]
                                # print(len(out_chunks), "3333")
                                out_chunks = torch.cat(out_chunks, 0)

                                # 填充数据，其他未参与细渲染计算的结果全部用默认值
                                out_rgb = torch.full((self.batch, self.fine_numb, 3), 1.0, device=self.device)
                                out_sigma = torch.full((self.batch, self.fine_numb, 1), self.sigma_default, device=self.device)
                                out_sh = torch.full((self.batch, self.fine_numb, dim_sh), 0.0, device=self.device)
                                out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)
                                out_defaults[idx_render_fine[:, 0], idx_render_fine[:, 1]] = out_chunks                     
                                
                                # 拆分信息
                                # sigmas: [self.batch, 128, 1]
                                # rgbs_pred: [self.batch, 128, 3]
                                # shs: [self.batch, 128, 27]
                                sigmas_pred, rgbs_fine, _ = torch.split(out_defaults, (1, 3, dim_sh), dim=-1) 
                                # 修正格式变为[self.batch, fine_numb]
                                sigmas_fine = sigmas_pred.squeeze(-1)

                                # ##############
                                # 6.获取fine_delta
                                # ##############
                                deltas_fine = z_vals_fine_all[:, 1:] - z_vals_fine_all[:, :-1]
                                delta_inf_fine = 1e10 * torch.ones_like(deltas_fine[:, :1])
                                # [self.batch, fine_numb] 
                                deltas_fine = torch.cat([deltas_fine, delta_inf_fine], -1) 

                                # ##############
                                # 7.sigma,delta查询weights_fine
                                # ##############
                                weights_fine = self.sigma2weights(deltas_fine, sigmas_fine)
                                weights_sum = weights_fine.sum(1) 

                                # ##############
                                # 8. 计算最终渲染颜色
                                # ##############
                                rgb_final = torch.sum(weights_fine.unsqueeze(-1)*rgbs_fine, -2)
                                # print(rgb_final.shape)

                                if self.white_back:
                                    rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

                            results_cat[cur_clss] += [rgb_final]
                            rgb_cat[cur_clss] += [rgbs_data[:, cur_clss, :]]
        
                # 合并计算结果
                for cur_clss in range(total_clss_numb):
                    results_cat[cur_clss] = torch.cat(results_cat[cur_clss], 0)
                    rgb_cat[cur_clss] = torch.cat(rgb_cat[cur_clss], 0)

                logging.info("保存图片中...")

            for cur_clss in range(total_clss_numb):
                cur_img_rays = results_cat[cur_clss][0:single_img_rays_numb]
                # 真值结果
                cur_img_gt = rgb_cat[cur_clss][0:single_img_rays_numb]

                img = cur_img_rays.view(self.img_wh, self.img_wh, 3).cpu()
                gt = cur_img_gt.view(self.img_wh, self.img_wh, 3).cpu()
                img = img.permute(2, 0, 1) # (3, H, W)
                gt = gt.permute(2, 0, 1) # (3, H, W)

                # 保存图片
                img_path = os.path.join(Path(self.train_img_pth), Path("epoch_"+ str(epoch) + "_" + str(cur_clss) + ".png"))
                gt_path = os.path.join(Path(self.train_img_pth), Path("epoch_"+ str(epoch) + "_" + str(cur_clss) + "_gt.png"))
                os.makedirs(os.path.dirname(img_path), exist_ok=True)

                transforms.ToPILImage()(img).convert("RGB").save(img_path)
                transforms.ToPILImage()(gt).convert("RGB").save(gt_path)

                psnr_coarse = psnr(img, gt)
                logging.info("{}_PSNR:{}".format(cur_clss, psnr_coarse))
                        
        # 其他线程等待验证线程完成
        if self.sys_args["distributed"]:
            dist.barrier()

        torch.cuda.empty_cache()

    def valid_rays(self, valid_rays_loader, epoch):
        # 首先清空缓存
        torch.cuda.empty_cache()
        rank = get_rank()
            
        # 分布式情况下用0线程卡测试
        # 单卡模式默认rank为0
        if rank == 0:
            val_model = NeRF(self.sys_args, type="fine").to(self.device)
            val_model_rays = NeRF_Rays(self.sys_args).to(self.device)
            val_model.eval()
            val_model_rays.eval()
            # 载入验证模型，生成验证结果
            logging.info("载入验证模型：{}".format(self.model_name))
            fine_ckpt = torch.load(self.file_path, map_location = self.device)["model"]

            # 编辑权重信息
            new_fine_ckpt = fine_ckpt.copy()
            new_rays_ckpt = fine_ckpt.copy()
            
            for key in fine_ckpt:
                # 删除粗查询的权重
                if (key.split(".")[0] != "nerf_fine") and (key.split(".")[0] != "nerf_rays"):
                    del(new_fine_ckpt[key])
                    del(new_rays_ckpt[key])
                # 重命名细粒度查询的相关信息
                elif (key.split(".")[0] == "nerf_fine"):
                    # 先删除
                    del(new_fine_ckpt[key])
                    del(new_rays_ckpt[key])
                    # 再写入
                    new_key_name = key.replace((key.split(".")[0] + "."), "")
                    new_fine_ckpt[new_key_name] = fine_ckpt[key]
                else:
                    # 先删除
                    del(new_fine_ckpt[key])
                    del(new_rays_ckpt[key])
                    # 再写入
                    new_key_name = key.replace((key.split(".")[0] + "."), "")
                    new_rays_ckpt[new_key_name] = fine_ckpt[key]

            # 载入权重
            val_model.load_state_dict(new_fine_ckpt)
            val_model_rays.load_state_dict(new_rays_ckpt)
            # 缓存结果
            results_cat = {}
            results_rays_cat = {}
            rgb_cat = {}
            # 单张图片需要的所有光线数量
            single_img_rays_numb = self.img_wh * self.img_wh
            numb_loader = round(single_img_rays_numb / self.batch + 0.5)

            # 前项测试
            with torch.no_grad():
                logging.info("渲染图片中...")

                for idx, data in enumerate(valid_rays_loader):
                    if idx > numb_loader:
                        break
                    else:
                        # 获取渲染光线，data:[Batch, W*H, 6]
                        rgbs_data, rays_data, clss_data = data
                        # GPU
                        rays_data = rays_data.to(self.device) #[Batch, numb_clss, 3]
                        rgbs_data = rgbs_data.to(self.device) #[Batch, numb_clss, 3]
                        clss_data = clss_data.to(self.device) #[Batch, numb_clss, 1]
                        # 总种类数量
                        total_clss_numb = rays_data.shape[1]
                        # 循环处理每个类别
                        for cur_clss in range(total_clss_numb):
                            if idx == 0:
                                results_cat[cur_clss] = []
                                rgb_cat[cur_clss] = []
                                results_rays_cat[cur_clss] = []
                            # ##############
                            # 1.获取sigma
                            # ##############
                            rays_o, rays_d = rays_data[:, cur_clss, 0:3], rays_data[:, cur_clss, 3:6] # both (self.batch, 3)
                            clss = clss_data[:,cur_clss,:]
                            if not self.skip_coarse:
                                # 将分份后的采样方法应用于每一条射线, [self.batch, 128]
                                z_vals_coarse_all = self.z_vals_coarse.clone().expand(self.batch, -1)
                                # 将初始化的标准无方向信息的光线采样转化为特定观察位置的采样信息
                                # [self.batch, 128, 3]
                                xyz_sampled_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1)*z_vals_coarse_all.unsqueeze(2)   
                                xyz_coarse = xyz_sampled_coarse.reshape(-1, 3)
                                # 集体查询，直接获取batch数量的sigma信息
                                # 查找对应粗采样位置的各个sigma数值
                                # [self.batch, 128]
                                sigmas_coarse = self.query_coarse_sigma(xyz_coarse, cur_clss).reshape(self.batch, self.coarse_numb)
                                # ##############
                                # 2.获取coarse_delta
                                # ##############
                                # 将光路上的delta分为128份，最后一份无限长
                                # [self.batch, 127]
                                deltas_coarse = z_vals_coarse_all[:, 1:] - z_vals_coarse_all[:, :-1]
                                delta_inf_coarse = 1e10 * torch.ones_like(deltas_coarse[:, :1]) 
                                # [self.batch, 128]
                                deltas_coarse = torch.cat([deltas_coarse, delta_inf_coarse], -1)   
                                # ##############
                                # 3.sigma,delta查询weights
                                # ##############
                                # [self.batch, 128]
                                weights_coarse = self.sigma2weights(deltas_coarse, sigmas_coarse)
                                # ##############
                                # 4.weights_coarse生成细渲染索引
                                # ##############
                                idx_render = torch.nonzero(weights_coarse >= min(self.weight_thresh, weights_coarse.max().item()))
      
                                # z_vals_fine数值为粗采样的sample_scale倍, 目前是128*5
                                # [self.batch, fine_numb]
                                z_vals_fine_all = self.z_vals_fine.clone().expand(self.batch, -1)
                                # [X, self.sample_scale, 2]
                                idx_render = idx_render.unsqueeze(1).expand(-1, self.sample_scale, -1)
                                idx_render_fine = idx_render.clone()
                            
                                # 索引信息转换，由粗渲染生成的索引转化为细渲染阶段索引
                                # idx_render格式中的最后一项表示第idx_render[..., 0]条光线的第idx_render[..., 1]段
                                # idx_render[..., 1]的范围是0~128，乘以sample_scale表示扩大5倍，将128采样变为128*5倍，索引间距也乘以了5倍
                                # idx_render[..., 1] * self.sample_scale 尺寸为 [X, 5]
                                # (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale) 尺寸为 [1, 5]
                                # 两者相加结果是[X, 5]
                                # 两者相加的含义是将复制了5份的索引间距更加细化，索引间距乘以5倍后中间的间距再细化为5份，复制的每一份分别对应细化的一个分度
                                idx_render_fine[..., 1] = idx_render[..., 1] * self.sample_scale + (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale)
                                # 再重新整合为索引格式，变为[X*sample_scale, 2]
                                idx_render_fine = idx_render_fine.reshape(-1, 2)
                                # 筛选有效光线
                                # [self.batch, 128*sample_scale, 3]
                                xyz_sampled_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine_all.unsqueeze(2) 
                            else:
                                z_vals_fine_all = self.z_vals_fine.clone().expand(self.batch, -1)
                                xyz_sampled_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine_all.unsqueeze(2)
                                xyz_fine_for_query = xyz_sampled_fine.reshape(-1, 3) 
                                idx_render_fine = self.query_fine_index(xyz_fine_for_query, cur_clss).reshape(self.batch, -1)
                                idx_render_fine = torch.nonzero(idx_render_fine)
                                
                            # ##############
                            # 5.计算fine_sigma, fine_rgbs
                            # ##############
                            dim_sh = self.fine_dim_sh
                            xyz_fine = xyz_sampled_fine[idx_render_fine[:, 0], idx_render_fine[:, 1]].reshape(-1, 3)
                            # 筛选有效角度
                            # [self.batch, 128*sample_scale, 3]
                            view_dir = rays_d.unsqueeze(1).expand(-1, self.fine_numb, -1)
                            ray_clss = clss.unsqueeze(1).expand(-1, self.fine_numb, -1)

                            view_dir = view_dir[idx_render_fine[:, 0], idx_render_fine[:, 1]] 
                            ray_clss = ray_clss[idx_render_fine[:, 0], idx_render_fine[:, 1]]                     
                            # 记录筛选后还剩信息的数量
                            cur_numb_part = xyz_fine.shape[0]
                            # 输出信息的缓存容器
                            out_chunks = []

                            # 对batch进行了合并, 将batch分成了128份，而渲染函数只能每次渲染一段，所以还需要循环
                            for i in range(0, cur_numb_part, self.chunk):
                                # 对输入信息编码，注意，方向信息没有编码
                                input_encode_xyz = self.embedding_xyz(xyz_fine[i:i+self.chunk])
                                input_encode_cls = self.embedding_cls(ray_clss[i:i+self.chunk])
                                input_dir = view_dir[i:i+self.chunk]
                                # 计算输出信息，输出信息中有:
                                # out = torch.cat([sigma, rgb, sh], -1)
                                nerf_model_out = val_model(input_encode_xyz, input_encode_cls, input_dir)
                                out_chunks += [nerf_model_out]

                            out_chunks = torch.cat(out_chunks, 0)
                            # 填充数据，其他未参与细渲染计算的结果全部用默认值
                            out_rgb = torch.full((self.batch, self.fine_numb, 3), 1.0, device=self.device)
                            out_sigma = torch.full((self.batch, self.fine_numb, 1), self.sigma_default, device=self.device)
                            out_sh = torch.full((self.batch, self.fine_numb, dim_sh), 0.0, device=self.device)
                            out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)
                            out_defaults[idx_render_fine[:, 0], idx_render_fine[:, 1]] = out_chunks                     
                            
                            # 拆分信息
                            # sigmas: [self.batch, 128, 1]
                            # rgbs_pred: [self.batch, 128, 3]
                            # shs: [self.batch, 128, 27]
                            sigmas_pred, rgbs_fine, _ = torch.split(out_defaults, (1, 3, dim_sh), dim=-1) 
                            # 修正格式变为[self.batch, fine_numb]
                            sigmas_fine = sigmas_pred.squeeze(-1)

                            # ##############
                            # 6.获取fine_delta
                            # ##############
                            deltas_fine = z_vals_fine_all[:, 1:] - z_vals_fine_all[:, :-1]
                            delta_inf_fine = 1e10 * torch.ones_like(deltas_fine[:, :1])
                            # [self.batch, fine_numb] 
                            deltas_fine = torch.cat([deltas_fine, delta_inf_fine], -1) 

                            # ##############
                            # 7.sigma,delta查询weights_fine
                            # ##############
                            weights_fine = self.sigma2weights(deltas_fine, sigmas_fine)
                            weights_sum = weights_fine.sum(1) 

                            # ##############
                            # 8. 计算最终渲染颜色
                            # ##############
                            rgb_final = torch.sum(weights_fine.unsqueeze(-1)*rgbs_fine, -2)

                            if self.white_back:
                                rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)
                            
                            if self.skip_coarse:
                                rays_encode_cls = self.embedding_cls(clss)
                                xyz_fine = xyz_sampled_fine.reshape(xyz_sampled_fine.shape[0], -1)
                                rays_encode_xyz = self.embedding_xyz(xyz_fine)
                                rgb_rays = val_model_rays(rays_encode_xyz, rays_encode_cls)
                                # rgb_rays = torch.sum(rgb_sample_rays, 1)
                                results_rays_cat[cur_clss] += [rgb_rays]
                                
                            results_cat[cur_clss] += [rgb_final]
                            rgb_cat[cur_clss] += [rgbs_data[:, cur_clss, :]]
                            
                # 合并计算结果
                for cur_clss in range(total_clss_numb):
                    results_cat[cur_clss] = torch.cat(results_cat[cur_clss], 0)
                    rgb_cat[cur_clss] = torch.cat(rgb_cat[cur_clss], 0)
                    if self.skip_coarse:
                        results_rays_cat[cur_clss] = torch.cat(results_rays_cat[cur_clss], 0)
                    
                logging.info("保存图片中...")

            for cur_clss in range(total_clss_numb):
                if self.skip_coarse:
                    cur_rays = results_rays_cat[cur_clss][0:single_img_rays_numb]
                    ray = cur_rays.view(self.img_wh, self.img_wh, 3).cpu()
                    ray = ray.permute(2, 0, 1) # (3, H, W)
                    ray_path = os.path.join(Path(self.train_img_pth), Path("epoch_"+ str(epoch) + "_" + str(cur_clss) + "_ray.png"))
                    transforms.ToPILImage()(ray).convert("RGB").save(ray_path)
                    
                cur_img_rays = results_cat[cur_clss][0:single_img_rays_numb]
                cur_img_gt = rgb_cat[cur_clss][0:single_img_rays_numb]

                img = cur_img_rays.view(self.img_wh, self.img_wh, 3).cpu()
                gt = cur_img_gt.view(self.img_wh, self.img_wh, 3).cpu()
                
                img = img.permute(2, 0, 1) # (3, H, W)
                gt = gt.permute(2, 0, 1) # (3, H, W)

                # 保存图片
                img_path = os.path.join(Path(self.train_img_pth), Path("epoch_"+ str(epoch) + "_" + str(cur_clss) + ".png"))
                gt_path = os.path.join(Path(self.train_img_pth), Path("epoch_"+ str(epoch) + "_" + str(cur_clss) + "_gt.png"))
                os.makedirs(os.path.dirname(img_path), exist_ok=True)

                transforms.ToPILImage()(img).convert("RGB").save(img_path)
                transforms.ToPILImage()(gt).convert("RGB").save(gt_path)

                psnr_img = psnr(img, gt)
                if self.skip_coarse:
                    psnr_ray = psnr(ray, gt)
                    logging.info("{}_PSNR_img:{}, {}_PSNR_ray:{}".format(cur_clss, psnr_img, cur_clss, psnr_ray))
                else:
                    logging.info("{}_PSNR_img:{}".format(cur_clss, psnr_img))
                    
        # 其他线程等待验证线程完成
        if self.sys_args["distributed"]:
            dist.barrier()

        torch.cuda.empty_cache()

    def save_model(self, model, sys_param, all_class_name):
        # 获取保存路径
        weights_pth = sys_param['root_weight']
        save_path = os.path.join(Path(weights_pth), Path("train"))
        # 判断一下路径是否存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sigma_voxels_coarse_clean = self.sigma_voxels_coarse.clone()
        sigma_voxels_coarse_clean[sigma_voxels_coarse_clean == self.sigma_init] = self.sigma_default

        # 确定当前保存的权重名称
        net  = "E-NeRF-all-"
        nowtime = time.strftime("%Y-%m-%d-%H-%M-%S.ckpt", time.localtime())
        # 最终保存文件名称
        self.model_name = net + nowtime
        # 最终保存文件路径
        self.file_path = os.path.join(Path(save_path), Path(self.model_name))

        save_dict = {'model': model.state_dict(), 'weight_voxels_fine': self.idx_voxels_fine, 'all_class_name': all_class_name}
        
        if sys_param["distributed"]:
            if dist.get_rank() == 0:
                torch.save(save_dict, self.file_path)
        else:
            torch.save(save_dict, self.file_path)
    
        logging.info('保存模型:{}'.format(self.model_name))
