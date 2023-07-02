import torch
import logging
import os
import json

import numpy as np

from torchvision import transforms as T
import torchvision.transforms.functional as transforms_F

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from model import get_ray_directions, get_rays
from model import FishEyeGenerator

class Data_enerf(torch.utils.data.Dataset):
    def __init__(self, system_param):
        # 参数转接
        self.system_param = system_param
        # 数据集存储的根目录
        self.data_root = self.system_param['data']
        # 获取设备信息
        self.device = self.system_param["device_type"]
        # 图片大小[800,800]
        self.img_wh = (self.system_param["default_res"], self.system_param["default_res"])
        # 定义图片转为tensor的具体操作
        self.define_transforms()
        # 数据容器
        self.cur_rgbs = []
        self.cur_rays = []
        self.cur_clss = []

        # 如果是训练模式，就读取数据
        if self.system_param["mode"] != 2:
            # 读取所有json文件内容，形成字典
            # 字典内容：
            # key: 当前数据集的名称，例如: lamborghini
            # key中内容：
            # cam_angle_x：当前数据集的相机视场角
            # class_id：类别的编号
            # all_pose：所有的位姿信息，列表格式:[Tensor1, Tensor2, ...]
            # all_image：所有的图片路径，列表格式:[PIL1, PIL2]
            self.all_class_name, self.all_class_info = self.load_all_json_info()
            class_numb = len(self.all_class_name)
            # 每个数据集应该具有的光线数量,100是每个类别数据集中图片的数量
            self.total_rays_numb = self.img_wh[0]*self.img_wh[1]*100
            # 合并所有数据
            for init_data_name in self.all_class_name:
                init_data_info = self.all_class_info[init_data_name]
                # 获取当前参与训练的模型名称
                cur_all_rgbs, cur_all_rays, cur_all_clss = self.train_data_input(init_data_name, init_data_info)
                # 合并信息，取公共数量部分的数据
                self.cur_rgbs += [cur_all_rgbs[:self.total_rays_numb, :]] 
                self.cur_rays += [cur_all_rays[:self.total_rays_numb, :]]  
                self.cur_clss += [cur_all_clss[:self.total_rays_numb, :]] 
            
            # [total_rays_numb, class_numb, 3]
            self.cur_rgbs = torch.stack(self.cur_rgbs, 1)
            # [total_rays_numb, class_numb, 6]
            self.cur_rays = torch.stack(self.cur_rays, 1)
            # [total_rays_numb, class_numb, 1]
            self.cur_clss = torch.stack(self.cur_clss, 1)
            
            logging.info("所有数据集准备完毕，共包含{}类目标".format(class_numb))

        # 测试模式需要调用test_pose_input函数
        else:
            # 是否使用单鱼眼模式
            self.fisheye_mode = self.system_param["fisheye_mode"]
            # 单鱼眼模式焦距
            self.fisheye_focal = self.system_param["fisheye_focal"]
            
    # 返回数据长度
    def __len__(self):
        # 训练、评估
        if self.system_param["mode"] != 2:
            return self.cur_rays.shape[0]
        # 测试模式
        else:
            return self.all_rays_test.shape[0]

    # 每轮迭代处理
    def __getitem__(self, idx):
        # 训练、评估
        if self.system_param["mode"] != 2:
            return self.cur_rgbs[idx], self.cur_rays[idx], self.cur_clss[idx]
        # 测试模式
        else:
            return self.all_rays_test[idx], self.all_clss_test[idx]      
    
    ###################核心函数###################
    # 载入所有类别数据中json文件的内容
    def load_all_json_info(self):
        logging.info("读取json文件数据中...")
        all_json_dict = {}     
        # 处理路径下所有的数据集
        all_data_name = os.listdir(self.data_root)
        all_data_name.sort()
        all_class_name = []
        # 处理.json
        for name in all_data_name:
            # 当前类别的json文件
            cur_train_json = os.path.join(Path(self.data_root), Path(name), Path("transforms_train.json"))         
            # 当前数据的类别名称
            cur_cls_name = name.split("_")[-1]
            all_class_name += [cur_cls_name]
            # 读取当前类别json文件的内容
            with open(cur_train_json, 'r') as f:
                js_info = json.load(f) 
            # 当前数据的视场角
            cur_cam_angle_rad = js_info["camera_angle_x"]
            # 当前数据的类别id
            cur_class_id = js_info["class"]
            # 当前数据的帧图像信息
            cur_all_pose = []
            cur_all_path = []
            # 对每一帧图像进行处理
            for frame in js_info['frames']:
                # 矩阵内容是相机坐标系到世界坐标系的转换矩阵(c2w)
                # 4x4
                pose = np.array(frame['transform_matrix'])
                # 转换当前pose信息为tensor
                c2w = torch.FloatTensor(pose)
                cur_all_pose += [c2w]
                # 获得当前图片的路径
                cur_img_path = os.path.join(Path(self.data_root), Path(name), Path(frame["file_path"] + ".png"))
                # cur_img_path = os.path.join(Path(self.data_root), Path(name), Path(frame["file_path"] + ".JPG"))
                cur_all_path += [cur_img_path]
            # 写入当前内容
            all_json_dict[cur_cls_name] = {"cam_angle_x":cur_cam_angle_rad, 
                                           "class_id":cur_class_id,
                                           "all_pose":cur_all_pose,
                                           "all_image":cur_all_path}
        logging.info("所有数据集列表:{}".format(all_class_name))
        return all_class_name, all_json_dict

    # 训练、评估模式下数据的载入函数，载入blender格式数据
    def train_data_input(self, cur_data_name, cur_data_info):
        logging.info("当前训练数据:{}".format(cur_data_name))
        # 拆分数据
        # 相机视场角，弧度制
        cur_cam_angle_rad = cur_data_info["cam_angle_x"]
        # 类别id
        cur_class_id = cur_data_info["class_id"]
        # 所有位姿tensor
        cur_all_pose = cur_data_info["all_pose"]
        # 所有图片路径
        cur_all_path = cur_data_info["all_image"]

        # 确认对应关系
        all_numb = len(cur_all_pose)
        assert all_numb == len(cur_all_path), "pose and image size are not matching!"

        # 容器初始化
        cur_all_rgbs = []        
        # 处理当前类别的数据
        # 处理所有rgb信息
        for tips_idx in range(all_numb):
            # 获取当前的idx
            cur_img_pth = cur_all_path[tips_idx]
            # 存储当前图片的路径       
            cur_img = Image.open(cur_img_pth)
            
            if cur_img.size != self.img_wh:
                cur_img = cur_img.resize(self.img_wh, Image.Resampling.LANCZOS)
            
            # 如果图片的设定长宽和读入图片的长宽有差异，就按照设定的尺寸resize图片           
            if self.img_wh[0] != cur_img.size[0]:
                cur_img = cur_img.resize(self.img_wh, Image.Resampling.LANCZOS)
            
            # 将图片转换成Tensor，映射到0-1，由于图片包含透明信息，所以是4通道，RGBA格式
            cur_img = self.transform(cur_img) # (4, h, w)
            
            if cur_img.shape[0] == 4:
                # 图片格式转换
                cur_img = cur_img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                # 将A通道融合进RGB通道
                # Target.R = ((1 - Source.A) * BGColor.R) + (Source.A * Source.R)
                # Target.G = ((1 - Source.A) * BGColor.G) + (Source.A * Source.G)
                # Target.B = ((1 - Source.A) * BGColor.B) + (Source.A * Source.B)
                cur_img = cur_img[:, :3]*cur_img[:, -1:] + (1-cur_img[:, -1:]) # blend A to RGB
            else:
                cur_img = cur_img.view(3, -1).permute(1, 0) # (h*w, 4) RGBA
                
            cur_all_rgbs += [cur_img]

        logging.info("合并读取数据中...")    
        # 合并所有的光线
        # [N*H*W, 3]
        cur_all_rgbs = torch.cat(cur_all_rgbs, 0)
        # 转换位姿信息为射线
        # [N, H*W, 6]->[N*W*H, 6]
        cur_all_rays = self.pose2rays(cur_cam_angle_rad, cur_all_pose).reshape(-1, 6) 
        # [N*H*W, 1]
        cur_all_cls = torch.ones(cur_all_rgbs.shape[0], 1)*cur_class_id
            
        # 输出格式检查
        assert cur_all_rays.shape[0] == cur_all_rgbs.shape[0], "rgbs and poses are not equal, check data generate stage!"                
                
        logging.info("当前训练数据准备完成！")

        return cur_all_rgbs, cur_all_rays, cur_all_cls

    # 测试模式下数据的载入函数，自定义theta\phi\radius
    def test_pose_input(self, theta, phi, radius, clss):
        assert theta.shape[0] == phi.shape[0] == radius.shape[0], 'pose shape must all equal!'
        single_img_rays = self.img_wh[0] * self.img_wh[1]
        # 图像的视场角度
        cam_angle_x = torch.Tensor([self.system_param["cam_fov"]]).float()
        # 水平, 垂直，相机视场角
        theta, phi, cam_angle_x = self.angle2rad(theta, phi, cam_angle_x)
        radius = torch.Tensor(radius).float()
        # 定义self.transform，转为tensor，范围[0,1]
        self.define_transforms()
        # 打印数据处理信息
        logging.info("输入位姿数据计算中...")
        # 生成c2w矩阵, [N, 4, 4]
        self.c2w_test = self.pose_spherical(phi, theta, radius)
        
        # 单鱼眼模式
        if self.fisheye_mode:
            # 定义鱼眼图像变换器
            gen_fish = FishEyeGenerator(self.fisheye_focal, self.img_wh)
            mask = np.ones(self.img_wh, dtype=np.uint8)
            all_rays_test = self.pose2fishrays(cam_angle_x, self.c2w_test)
            gen_fish._calc_cord_map(mask)
            map_rows = np.clip(gen_fish._map_rows, 0, self.img_wh[1]-1) 
            map_cols = np.clip(gen_fish._map_cols, 0, self.img_wh[0]-1)
            self.all_rays_test = all_rays_test[:, map_rows, map_cols, :]
        else:
            self.all_rays_test = self.pose2rays(cam_angle_x, self.c2w_test)

        # [Batch, 1]
        self.all_clss_test = torch.Tensor(clss).reshape([-1,1]).expand(-1, single_img_rays).reshape([-1, 1])
        # [Batch, 6]
        self.all_rays_test = self.all_rays_test.reshape(-1, 6)

        logging.info("姿数据计算完毕！")

    ###################功能函数###################
    # 角度转成弧度信息
    def angle2rad(self, theta, phi, cam_angle_x):
        theta_rad = torch.Tensor(theta/180.*np.pi)
        phi_rad = torch.Tensor(phi/180.*np.pi)*(-1)
        cam_angle_x = torch.Tensor(cam_angle_x/180.*np.pi)
        return theta_rad, phi_rad, cam_angle_x

    # 弧度转成旋转矩阵
    def pose_spherical(self, phi, theta, radius):
        # 获取数据数量
        pose_len = theta.shape[0]
        # 计算基本元素
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # 平移矩阵
        diag_trans_t = torch.ones((4), dtype=torch.float)
        trans_t = torch.diag_embed(diag_trans_t).repeat((pose_len,1,1))
        trans_t[:,2,-1] = radius
        # 全1变量tensor
        rot_one = torch.ones((pose_len), dtype=torch.float)
        # 旋转矩阵phi
        rot_phi = torch.zeros((pose_len, 4, 4),dtype=torch.float)
        rot_phi[:,0,0] = rot_one
        rot_phi[:,-1,-1] = rot_one
        rot_phi[:,1,1] = cos_phi
        rot_phi[:,1,2] = sin_phi*(-1)
        rot_phi[:,2,1] = sin_phi
        rot_phi[:,2,2] = cos_phi  
        # 旋转矩阵theta
        rot_theta = torch.zeros((pose_len, 4, 4),dtype=torch.float)
        rot_theta[:,0,0] = cos_theta
        rot_theta[:,0,2] = sin_theta*(-1)
        rot_theta[:,1,1] = rot_one
        rot_theta[:,2,0] = sin_theta
        rot_theta[:,2,2] = cos_theta
        rot_theta[:,-1,-1] = rot_one              
        # 合成位姿矩阵
        c2w = trans_t
        c2w = rot_phi @ c2w
        c2w = rot_theta @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        
        return c2w

    # 定义数据转换模型，直接变为tensor，范围[0,1]
    def define_transforms(self):
        self.transform = T.ToTensor()  

    # 将旋转矩阵生成rays信息
    def pose2rays(self, cam_angle_x, c2w):
        # 最终所有rays和rgb的缓存容器
        all_rays = []
        # 图像长宽
        w, h = self.img_wh
        # 计算焦距，camera_angle_x表示相机水平方向的视场角
        # json文件中的角度是弧度制
        focal = 0.5*self.img_wh[0]/np.tan(0.5*cam_angle_x) # original focal length
        # 返回图像中各个像素与相机中心的夹角正切值
        # 一共三个通道：
        # 第一个是x方向，当前像素光线向量的x方向分量
        # 第二个是y方向，当前像素光线向量的y方向分量
        # 第三个是z方向，当z=1时，前面的tan(x) = x
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        for pose in c2w:
            pose = pose[:3, :4]
            
            # 转换角度到世界坐标系，获得世界坐标系下的光线原点及光线方向
            # rays_o尺寸(h*w, 3)，每个元素表示[x,y,z]世界坐标系下的相机原点位置
            # rays_d尺寸(h*w, 3)，每个元素表示[thetax,thetay,thetaz]，与相机中心轴在各个方向的夹角，弧度制
            rays_o, rays_d = get_rays(directions, pose, img_dim = False) # both (h*w, 3)
            # 转换为单位向量
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            # 将每一帧图像的世界坐标系光线方向及原点信息进行存储
            all_rays += [torch.cat([rays_o, rays_d], 1)] # (h*w, 6)

        # [N, H*W, 6]
        all_rays = torch.stack(all_rays, 0)
        # print(all_rays, all_rays.shape, "all_raysall_raysall_raysall_raysall_raysall_raysall_raysall_raysall_rays")

        return all_rays

    # 将旋转矩阵生成rays信息
    def pose2fishrays(self, cam_angle_x, c2w):
        # 最终所有rays和rgb的缓存容器
        all_rays = []
        # 图像长宽
        w, h = self.img_wh
        # 计算焦距，camera_angle_x表示相机水平方向的视场角
        # json文件中的角度是弧度制
        focal = 0.5*640/np.tan(0.5*cam_angle_x) # original focal length
        # 返回图像中各个像素与相机中心的夹角正切值
        # 一共三个通道：
        # 第一个是x方向，当前像素光线向量的x方向分量
        # 第二个是y方向，当前像素光线向量的y方向分量
        # 第三个是z方向，当z=1时，前面的tan(x) = x
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        for pose in c2w:
            pose = pose[:3, :4]
            rays_o, rays_d = get_rays(directions, pose, img_dim = True) # both (h*w, 3)        
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            all_rays += [torch.cat([rays_o, rays_d], 2)] # (h*w, 6)

        # [N, H*W, 6]
        all_rays = torch.stack(all_rays, 0)
        # print(all_rays, all_rays.shape, "all_raysall_raysall_raysall_raysall_raysall_raysall_raysall_raysall_rays")

        return all_rays

# 将dataset转换成dataloader
class Data_loader_enerf():
    def __init__(self, dataset, sys_param):
        self.dataset = dataset
        self.sys_param = sys_param
        # batch信息
        self.batch = self.sys_param["batch"]
        # val只用单卡，所以不用分布式
        self.sampler_no_shuffle = torch.utils.data.SequentialSampler(dataset)

        # 处理分布式情况下的数据集，仅用在训练上
        if sys_param['distributed']:
            # shuffle乱序训练合成不了一张图片
            self.sampler = DistributedSampler(dataset, shuffle=True)
        else:
            self.sampler = torch.utils.data.RandomSampler(dataset)

        self.dataloader = self.pkg_dataloader()

    def pkg_dataloader(self):
        self.batch_sampler_train = torch.utils.data.BatchSampler(self.sampler, self.batch, drop_last=True)       
        self.batch_sampler_val   = torch.utils.data.BatchSampler(self.sampler_no_shuffle, self.batch, drop_last=False)
        
        loader_train = DataLoader(self.dataset,
                          batch_sampler=self.batch_sampler_train,
                          num_workers=32,
                          pin_memory=True)

        loader_val = DataLoader(self.dataset,
                          batch_sampler=self.batch_sampler_val,
                          num_workers=32,
                          pin_memory=True)        
        
        return {"train_loader": loader_train, "val_loader": loader_val}

