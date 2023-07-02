import torch
import os
import numpy as np
import argparse

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataload import Load_config

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

pt_file = "/home/gaoyu/CVPR2023_No_Transformer/weights/train/E-NeRF-all-2023-01-31-10-42-05.ckpt"

voxels_index = torch.load(pt_file, map_location = "cpu")["weight_voxels_fine"]

# space = Create_Unite_Space(sys_param)
# voxels_index = np.array(space.synthesis_block.cpu())

all_fig = plt.figure()
ax = Axes3D(all_fig, auto_add_to_figure=False)
all_fig.add_axes(ax)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
# ax.set_xlim(0, 384)
# ax.set_ylim(0, 384)
# ax.set_zlim(0, 384)
plt.gca().set_box_aspect((1, 1, 1))

voxel_scale = voxels_index.shape[-1]
step = 2

axis_range = np.arange(0, voxel_scale, step)
x_grid, y_grid = np.meshgrid(axis_range, axis_range)

cur_voxels = voxels_index[0, ...]

# 绘制384边界
idx_ptx_384= [0, 384, 384, 0, 0, 0, 384, 384, 0, 0]
idx_pty_384= [0, 0, 384, 384, 0, 0, 0, 384, 384, 0]
idx_ptz_384= [0, 0, 0, 0, 0, 384, 384, 384, 384, 384]

ax.plot(idx_ptx_384, idx_pty_384, idx_ptz_384, c='b')
test_cur_voxels = np.zeros_like(cur_voxels, dtype=np.bool8)
test_cur_voxels[150:250, 150:250, 150:250] = cur_voxels[150:250, 150:250, 150:250]
cur_voxels = test_cur_voxels

# 三维表示图
for z in np.arange(0, voxel_scale, step):
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)
    z_grid = np.ones_like(y_grid)*z
    cur_slice = np.array(cur_voxels[::step, ::step, z]).reshape(-1).astype(np.float32)

    ax.scatter(xs=x_grid, ys=y_grid, zs=z_grid, c='r', s=cur_slice, alpha=1, marker='.')

# 二维表示图
# for z in np.arange(0, voxel_scale, step):
#     x_grid = np.arange(0, voxel_scale, step)
#     y_grid = np.ones_like(x_grid)*voxel_scale/2
#     z_grid = np.ones_like(y_grid)*z
#     cur_line = np.array(cur_voxels[int(voxel_scale/2), ::step, z]).reshape(-1).astype(np.float32)

#     ax.scatter(xs=x_grid, ys=y_grid, zs=z_grid, c='r', s=cur_line, alpha=1, marker='.')

plt.show()