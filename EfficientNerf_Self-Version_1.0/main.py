import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from dataload import Data_enerf, Data_loader_enerf, Load_config
from model import Efficient_NeEF_Loss, Efficient_NeRF, RAdam, get_rank
from utils import Tensorboard_config

def efficient_nerf_train(sys_param, dataset, enerf_model, tblogger):
    # 获取设备类型,cuda
    device = sys_param["device_type"]
    # 学习速率
    learning_rate = sys_param["global_lr"]
    # 训练的总数量
    train_epoch = sys_param["train_epoch"]
    # 获取所有类别的名称
    all_class_name = dataset.all_class_name    
    # 建立dataloader
    loader_enerf = Data_loader_enerf(dataset, sys_param)       
    # 获取数据集
    sampler_train = loader_enerf.sampler
    dataloader = loader_enerf.dataloader["train_loader"]
    # 验证数据集
    valloader = loader_enerf.dataloader["val_loader"]      
    
    # 分布式判定
    if sys_param["distributed"]:
        # 将模型放到不同的gpu里
        enerf_model = torch.nn.parallel.DistributedDataParallel(enerf_model.to(device), device_ids=[sys_param['gpu']], find_unused_parameters=True)
        enerf_model_without_ddp = enerf_model.module
    else:
        enerf_model.to(device)
        # 先保存未进行分布式部署的网络，因为模型参数需要从原始模型里摘取
        enerf_model_without_ddp = enerf_model    

    # 测试结果保存路径
    save_pth = sys_param["demo_render_pth"] 
    # 定义损失函数
    loss_func = Efficient_NeEF_Loss(sys_param).to(device)
    # 优化器例化
    optimizer = RAdam(enerf_model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=sys_param["weight_d"])
    # optimizer = torch.optim.Adam(enerf_model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=sys_param["weight_d"])  
    # 训练
    enerf_model.train()

    # 文件夹创建及准备
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    
    current_step = 0
    extract_epoch = None
    # 开始训练
    for epoch in range(train_epoch):
        # 损失累加量
        running_loss = 0

        if sys_param["distributed"]:
            # 每个epoch开始的时候需要设置一下epoch
            # 这样做的目的是保证每个训练周期shuffle的序列不一样
            # 如果不设置，则每个周期shuffle的顺序都是相同的
            sampler_train.set_epoch(train_epoch)

        if extract_epoch is not None:
            enerf_model_without_ddp.skip_coarse = True
            logging.info("网络微调中...")
            
        # 提取index_render的轮数
        if (epoch >= (train_epoch/2)) and (enerf_model_without_ddp.skip_coarse == False):
            enerf_model_without_ddp.extract_time = True
            extract_epoch = epoch
            logging.info("提取占位索引块数开始...")
        
        # 配置训练显示进度条   
        with tqdm(total = len(dataloader), 
                desc='Train_Epoch:{}'.format(epoch), 
                bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]', 
                ncols=150) as bar:
            for step, data in enumerate(dataloader):
                current_step += 1
                # 梯度清零
                optimizer.zero_grad()
                # rgb:  [Batch, cls_numb, 3]
                # rays: [Batch, cls_numb, 6]
                # cls:  [Batch, cls_numb, 1]
                rgbs, rays, clss = data
                # print(rays, "teetetetetetetett") 00l 
                # print("\n", "输入光线:", rays)
                # 前项传播计算
                results = enerf_model(rays.to(device), clss.to(device))                
                # 计算前项损失
                loss_total = loss_func(results, rgbs.to(device))
                # 增加监控变量
                if tblogger is not None:
                    tblogger.add_scalar("TOTAL/Loss_Train", loss_total, current_step)
                # loss反向传播
                loss_total.backward()
                # 更新梯度信息
                optimizer.step()
                # 计算平均损失
                running_loss += loss_total.item()
                ave_loss = running_loss/(step + 1)

                bar.set_postfix_str('AveLoss:{:^7.5f}'.format(ave_loss))
                bar.update()

        enerf_model_without_ddp.save_model(enerf_model_without_ddp, sys_param, all_class_name)
        # 验证模型
        enerf_model_without_ddp.valid_fine(valloader, epoch)
        torch.cuda.empty_cache()

# Effcient Nerf前项测试函数
@torch.no_grad()
def efficient_nerf_test(sys_param, loader, enerf_model, phi_step=None):
    # 获取设备类型,cuda
    device = sys_param["device_type"]
    # 获取数据集
    dataloader = loader.dataloader["val_loader"]
    # 将模型放在gpu上
    enerf_model.to(device)
    # 评估模式
    enerf_model.eval()
    # 图片大小
    img_h, img_w = sys_param["default_res"], sys_param["default_res"]
    # 测试结果保存路径
    nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_pth = os.path.join(Path(sys_param["demo_render_pth"] + "_" + nowtime))
    # 创建路径
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    results_cat = []
    with tqdm(total = len(dataloader), ncols=70) as t:
        for i, data in enumerate(dataloader):
            # 获取渲染光线，data:[Batch, W*H, 6]
            rays, clss = data
            # 前项传播计算
            results = enerf_model(rays.to(device), clss.to(device))
            results_cat += [results]
            t.update()
    
    # 合并所有光线
    results_cat = torch.cat(results_cat, 0)
    img_chunk = img_h*img_w
    img_name_idx = 0

    # 恢复图像并保存
    for i in range(0, results_cat.shape[0], img_chunk):
        cur_img_rays = results_cat[i:i+img_chunk]
        img = cur_img_rays.view(img_h, img_w, 3).cpu()
        img = img.permute(2, 0, 1) # (3, H, W)
        
        if phi_step is not None:
            img_name = "C02_" + str(i).zfill(3) + "_" + str(phi_step).zfill(2) + ".png"
        else:
            img_name = str(img_name_idx).zfill(4) + ".png"
            
        img_pth = os.path.join(Path(save_pth), Path(img_name))
        # 转换图片格式
        transforms.ToPILImage()(img).convert("RGB").save(img_pth)
        img_name_idx += 1

    torch.cuda.empty_cache()

@torch.no_grad()
def efficient_nerf_test_single_img(sys_param, loader, enerf_model, phi_step=None):
    # 获取设备类型,cuda
    device = sys_param["device_type"]
    # 获取数据集
    dataloader = loader.dataloader["val_loader"]
    # 将模型放在gpu上
    enerf_model.to(device)
    # 评估模式
    enerf_model.eval()
    # 图片大小
    img_h, img_w = sys_param["default_res"], sys_param["default_res"]
    # 测试结果保存路径
    nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_pth = os.path.join(Path(sys_param["demo_render_pth"] + "_" + nowtime))
    # 创建路径
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    img_name_idx = 0
    img_chunk = img_h*img_w
    with tqdm(total = len(dataloader), ncols=70) as t:
        for i, data in enumerate(dataloader):
            # 获取渲染光线，data:[Batch, W*H, 6]
            rays, clss = data
            # 前项传播计算
            results = enerf_model(rays.to(device), clss.to(device))
            # 合并
            if i == 0:
                results_cat = results
            else:
                # 合并光线
                results_cat = torch.cat([results_cat, results], 0)
                # 保存图像
                if results_cat.shape[0] > img_chunk:
                    cur_img_rays = results_cat[:img_chunk]
                    img = cur_img_rays.view(img_h, img_w, 3).cpu()
                    img = img.permute(2, 0, 1) # (3, H, W)
                    
                    if phi_step is not None:
                        img_name = "C02_" + str(i).zfill(3) + "_" + str(phi_step).zfill(2) + ".png"
                    else:
                        img_name = str(img_name_idx).zfill(4) + ".png" 
                    img_pth = os.path.join(Path(save_pth), Path(img_name))
                    # 转换图片格式
                    transforms.ToPILImage()(img).convert("RGB").save(img_pth)
                    img_name_idx += 1
                    # 扔掉前面的光线信息
                    results_cat = results_cat[img_chunk:]
                    logging.info("保存图像:{} \n".format(img_name))

            t.update()

    torch.cuda.empty_cache()


if __name__ == "__main__":    
    # 添加执行代码可调用的参数命令
    parser = argparse.ArgumentParser()
    
    # 可选项：配置文件的根目录，默认是当前工作目录下的config文件夹
    parser.add_argument('--config', type=str, default="./config",
                        help='root path of config file')
    # 可选项：测试、评估、训练模式
    parser.add_argument('--demo_enerf', action='store_true',
                        help='enerf rendering forward with test mode')

    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--train', action='store_true',
                        help='train mode')
    # 可选项：是否存储log文件
    parser.add_argument('--log', action='store_true',
                        help='save log information to log.txt file')
    # 可选项：是否存储log文件
    parser.add_argument('--start_device', type=int, default=0,
                        help='start training device for distributed mode')
    # 可选项：是否采用tensorboard进行可视化训练
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard tools to show training results')
    args = parser.parse_args()

    ##############
    # 开始执行
    ##############
    # 示例指令测试 python main.py --demo_enerf
    # 示例指令训练 python -m torch.distributed.launch --nproc_per_node=5 --use_env main.py --train --start_device 1
    
    # 读取配置信息
    config_info = Load_config(args)
    # 拆成各个部分的配置信息
    sys_param = config_info.system_info
    logging.info("系统参数:\n {} \n".format(sys_param))
    # 固定seed方便测试
    seed = sys_param["seed"] + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 训练模式
    if sys_param["mode"] == 0:
        # 初始化数据
        tblogger = Tensorboard_config(sys_param).tblogger
        # 初始化数据集 
        dataset_enerf = Data_enerf(sys_param)   
        # 例化网络模型
        enerf_model = Efficient_NeRF(sys_param)
        efficient_nerf_train(sys_param, dataset_enerf, enerf_model, tblogger)

    # 评估模式
    elif sys_param["mode"] == 1:
        pass
    # enerf测试模式
    elif sys_param["mode"] == 2:
        # 指定观察视角:水平、垂直、观察半径
        # theta = np.array([i for i in np.arange(0,360,1)]) 
        # phi = np.array([j for j in np.arange(0,90,0.25)]) 
        # radius = np.array([2.5 for k in np.arange(0,360,1)])
        # obj_cls = np.array([4 for k in np.arange(0,360,1)])
   
        theta = np.array([0, 5, 90]) 
        phi = np.array([0, 45, 30]) 
        radius = np.array([3, 3, 3])
        obj_cls = np.array([0, 0, 0])
        
        # 对输入数据进行从小到大排序
        # s2b = np.argsort(obj_cls)
        # theta = theta[s2b]
        # phi = phi[s2b]
        # radius = radius[s2b]
        # obj_cls = obj_cls[s2b]

        # 初始化数据集
        dataset_enerf = Data_enerf(sys_param)
        dataset_enerf.test_pose_input(theta, phi, radius, obj_cls)
        loader_enerf = Data_loader_enerf(dataset_enerf, sys_param)
        
        # 例化网络模型
        enerf_model = Efficient_NeRF(sys_param)
        efficient_nerf_test(sys_param, loader_enerf, enerf_model, phi_step=None)
        # efficient_nerf_test_single_img(sys_param, loader_enerf, enerf_model, phi_step=None)
# 版本备注：
# 测试把sigma体素块换成布尔体素块