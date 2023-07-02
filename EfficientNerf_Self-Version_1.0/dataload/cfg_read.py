from pathlib import Path
from utils import Log_config
from utils import Distributed_config

import yaml
import os
import logging

# 定义读取所有yaml文件的函数
class Load_config():
    def __init__(self, args):
        # 读取配置文件根目录
        self.root_config = args.config
        # 读取终端输入的信息【更新终端输入的信息】
        self.system_info, self.detr_info, self.e_nerf_info = self.load_terminal(args)
        # 读取yaml文件信息，并保存成三个字典【更新yaml文件中的信息】    
        self.load_yaml()
        # 初始化log
        Log_config(self.system_info)   
        # 初始化多卡训练模式【更新多卡配置信息】
        Distributed_config(self.system_info) 

    # 读取yaml文件中的各种信息
    def load_yaml(self):
        # 拼接各种配置路径
        path_yaml = os.path.join(Path(self.root_config), Path('enerf.yaml'))

        # 读取config文件,以字典的形式存于config
        with open(path_yaml, 'r', encoding='utf-8') as f:
            cfg_info = yaml.load(f, Loader = yaml.FullLoader)
            
            ################# 模式共用参数 #####################
            # 数据集路径
            self.data_root = cfg_info['system']['data']['root_data']
            # 默认输入和渲染图像的分辨率
            self.system_info['default_res'] = cfg_info['model']['render']['img_size']
            # 读取log文件存储路径
            self.system_info['log_pth'] = cfg_info['system']['log_params']['logpath']
            # 读取权重文件的存放、读取根目录
            self.system_info['root_weight'] = cfg_info['system']['weights_params']['root_weights']
            # 网络输出结果的存放根目录，不包括权重文件，权重文件在root_weight中
            self.system_info['root_out'] = cfg_info['system']['out_params']['root_out']
            # 读取设备信息
            self.system_info['device_type'] = cfg_info['system']['device']['dev']  
            self.system_info['device_id'] = cfg_info['system']['device']['id']  
            # 种子数值
            self.system_info['seed'] = cfg_info['system']['data']['seed'] 
            # tensorboard存放、读取目录
            self.system_info['tb_pth'] = cfg_info['system']['tensorboard_params']['tb_pth'] 
            # tensorboard每次运行是否删除之前的tensorboard文件
            self.system_info['tb_del'] = cfg_info['system']['tensorboard_params']['del_mode'] 

            ############# enerf配置参数 ##################
            # 相机视场角
            self.system_info['cam_fov'] = cfg_info['model']['camera']['camera_angle_x']
           
            # 渲染光线粗采样数量
            self.system_info['coarse_numb'] = cfg_info['model']['render']['N_samples_coarse']
            # 渲染光线细采样倍数，细采样是粗采样的倍数个
            self.system_info['sample_fine_times'] = cfg_info['model']['render']['N_samples_fine_times_coarse']

            # 渲染光线最近有效范围，采样开始的最近距离
            self.system_info['near'] = cfg_info['model']['render']['near']
            # 渲染光线最远有效范围，采样开始的最远距离
            self.system_info['far'] = cfg_info['model']['render']['far']
            
            # 渲染空间的粗分割数量
            self.system_info['grid_coarse'] = cfg_info['model']['render']['grid_coarse']
            # 渲染光线细采分割倍数，细分割是粗分割数量的倍数个
            self.system_info['grid_fine_times'] = cfg_info['model']['render']['grid_fine_times_coarse']

            # 粗采样生成权重的阈值，超过阈值的部分继续进行渲染
            self.system_info['coarse_weight_thresh'] = cfg_info['model']['render']['weight_t']
            # 渲染过程背景是否渲染为纯白色
            self.system_info['white_back'] = cfg_info['model']['render']['white_back']
            # 渲染图像的坐标边界
            self.system_info['global_boader_min'] = cfg_info['model']['render']['global_boader_min']
            self.system_info['global_boader_max'] = cfg_info['model']['render']['global_boader_max']
            # 初始不透明度
            self.system_info['sigma_init'] = cfg_info['model']['render']['sigma_init']
            # 默认不透明度
            self.system_info['sigma_default'] = cfg_info['model']['render']['sigma_default']
            # coarse调整阶段全局调整轮数
            self.system_info['warmup_step'] = cfg_info['model']['render']['warmup_step']
            # 位姿生成数据的范围
            self.system_info['theta_range'] = cfg_info['model']['render']['theta_max']
            self.system_info['phi_range'] = cfg_info['model']['render']['phi_max']
            self.system_info['radius_length'] = cfg_info['model']['render']['radius']
            # 将角度的回归映射到0-pose_angle_norm之间
            self.system_info["pose_angle_norm"] = cfg_info['model']['render']['pose_angle_norm']
            # 粗渲染阶段的beta系数，用来更新sigma存储库，可以理解为sigma存储库的学习速率
            self.system_info["coarse_beta"] = cfg_info['model']['render']['coarse_beta'] 
            # 损失函数
            self.system_info["loss_type"] = cfg_info['model']['loss']['type'] 
            # chunk渲染
            self.system_info["chunk"] = cfg_info['model']['render']['chunk'] 

            ############# 基础nerf配置参数 ##################
            # 输入光线位置信息进nerf渲染编码时生成的额外编码数量
            self.system_info["emb_freqs_xyz"] = cfg_info['model']['nerf']['emb_freqs_xyz']
            # 输入光线方向信息进nerf渲染编码时生成的额外编码数量
            self.system_info["emb_freqs_cls"] = cfg_info['model']['nerf']['emb_freqs_cls']
            
            # MLP网络深度
            self.system_info['coarse_MLP_depth'] = cfg_info['model']['nerf']['coarse_MLP_depth']
            # MLP网络每层的宽度
            self.system_info['coarse_MLP_width'] = cfg_info['model']['nerf']['coarse_MLP_width']
            # MLP网络哪一层进行跳跃链接
            self.system_info['coarse_MLP_skip'] = cfg_info['model']['nerf']['coarse_MLP_skip']
            # 球谐函数的阶数，阶数越高，描述光照变化情况越准确
            self.system_info["coarse_MLP_deg"] = cfg_info['model']['nerf']['coarse_MLP_deg']

            # MLP网络深度
            self.system_info['fine_MLP_depth'] = cfg_info['model']['nerf']['fine_MLP_depth']
            # MLP网络每层的宽度
            self.system_info['fine_MLP_width'] = cfg_info['model']['nerf']['fine_MLP_width']
            # MLP网络哪一层进行跳跃链接
            self.system_info['fine_MLP_skip'] = cfg_info['model']['nerf']['fine_MLP_skip']
            # 球谐函数的阶数，阶数越高，描述光照变化情况越准确
            self.system_info["fine_MLP_deg"] = cfg_info['model']['nerf']['fine_MLP_deg']
            
            # 读取enerf测试结果图片的保存路径
            self.system_info["demo_render_pth"] = os.path.join(Path(self.system_info['root_out']), 
                                                               Path(cfg_info['model']['render']['test_enerf_pth']))

            ################按模式读取########################
            # 训练模式
            if self.system_info['mode'] == 0:
                # 训练用图片路径
                self.system_info['data'] = self.data_root
                # 训练用batch
                self.system_info['batch'] = cfg_info['system']['train_params']['batch_size']
                # 训练用epoch
                self.system_info["train_epoch"] = cfg_info['system']['train_params']['train_epoch']
                # 全局学习速率
                self.system_info["global_lr"] = cfg_info['system']['train_params']['learning_rate']
                # 权重衰减系数
                self.system_info["weight_d"] = cfg_info['system']['train_params']['weight_decay']
            # 评估模式
            elif self.system_info['mode'] == 1:
                pass            
            # 测试模式
            else:
                # 测试用图片路径
                self.system_info['data'] = os.path.join(Path(self.data_root), Path("test"))   
                # 测试用batch
                self.system_info['batch'] = cfg_info['system']['test_params']['batch_size']
                # 测试用权重
                self.system_info["nerf_model_name"] = cfg_info['system']['test_params']['nerf_model_name']   
                # 测试是否使用鱼眼模式
                self.system_info["fisheye_mode"] = cfg_info['system']['test_params']['fisheye_mode']
                # 前项渲染鱼眼焦距
                self.system_info["fisheye_focal"] = cfg_info['system']['test_params']['fisheye_focal']
                # 合并模型的根目录
                self.system_info["unite_root"] = cfg_info['unite']['data']['ckpt_model_data']
                # 合并模型具体的名称
                self.system_info["ckpt_name"] = cfg_info['unite']['data']['ckpt_name']
                
    # 将终端输入的信息进行合并存储
    def load_terminal(self, args):
        # 定义系统字典
        system_info = {}
        detr_info = {}
        e_nerf_info = {}
        # 处理当前的运行模式, 按照顺序只读取第一个有效的
        for mode, flag in enumerate([args.train, args.eval, args.demo_enerf]):
            if flag is True:
                system_info['mode'] = mode
                break
        # 存储log信息
        system_info['log'] = args.log
        # 分布式训练过程开始设备
        system_info['start_device'] = args.start_device    
        # 是否使用tensorboard
        system_info['tb_available'] = args.tensorboard

        return system_info, detr_info, e_nerf_info