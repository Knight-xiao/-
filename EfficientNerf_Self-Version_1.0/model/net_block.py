import torch
import torch.nn as nn

from .net_utils import eval_sh

# e_nerf编码,将位置和方向进行sin\cos编码
class Embedding(nn.Module):
    def __init__(self, N_freqs):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        # 除了原本的信息x,额外生成的编码信息数量
        self.N_freqs = N_freqs
        # 输入通道数量，图片是RGB，即为3
        self.in_channels = 3
        self.funcs = [torch.sin, torch.cos]
        # 每个通道原本有3个，加上两套sin,cos编码，频率是10，编码数量就是3*2*10=60
        # 一共就是3+60 = 63
        self.out_channels = self.in_channels*(len(self.funcs)*N_freqs+1)
        # 对数模式
        self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)

    def forward(self, x):
        out = [x] 
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

# NERF网络
class NeRF(nn.Module):
    def __init__(self, enerf_args, type="coarse"):
        super(NeRF, self).__init__()
        # 从embedding出来的位置编码维度：RGB通道*(sin和cos * 编码频率 + 1)
        # +1是将未编码的原始位置加入到最终的编码结果中
        self.in_channels_xyz = 3*(2*enerf_args["emb_freqs_xyz"] + 1) #63
        # 与位置编码同理 
        self.in_channels_cls = (2*enerf_args["emb_freqs_cls"] + 1) #9
        # 合并输入
        self.in_channels_all = self.in_channels_xyz + self.in_channels_cls
        
        # 判定模式，粗细调整网络模型的结构不同
        if type == "coarse":                
            # Nerf核心编码网络的深度，理解为MLP有多少层
            self.depth = enerf_args["coarse_MLP_depth"]
            # 编码网络每一层的宽度，理解为MLP每一层神经元数量
            self.width = enerf_args["coarse_MLP_width"]

            # 跳跃链接层，列表形式，指定具体层数进行跳跃链接
            self.skips = enerf_args["coarse_MLP_skip"]
            # 球谐函数的阶数，阶数越高，描述光照变化情况越准确
            # 一般描述变化比较平滑的环境漫反射部分，用3阶SH就足够了
            self.deg = enerf_args["coarse_MLP_deg"]
        
        elif type == "fine":                
            # Nerf核心编码网络的深度，理解为MLP有多少层
            self.depth = enerf_args["fine_MLP_depth"]
            # 编码网络每一层的宽度，理解为MLP每一层神经元数量
            self.width = enerf_args["fine_MLP_width"]

            # 跳跃链接层，列表形式，指定具体层数进行跳跃链接
            self.skips = enerf_args["fine_MLP_skip"]
            # 球谐函数的阶数，阶数越高，描述光照变化情况越准确
            # 一般描述变化比较平滑的环境漫反射部分，用3阶SH就足够了
            self.deg = enerf_args["fine_MLP_deg"]

        # xyz编码网络层进行构建
        for i in range(self.depth):
            # 第一层直接线性链接
            if i == 0:
                # 网络输入神经元数量是in_channels_all
                layer = nn.Linear(self.in_channels_all, self.width)
            # 跳跃层输入神经元数量需要加上in_channels_xyz
            elif i in self.skips:
                # 多出in_channels_all个神经元
                layer = nn.Linear(self.width + self.in_channels_all, self.width)
            else:
                # 其余层直接隐含层，W对W链接
                layer = nn.Linear(self.width, self.width)
            # 每一层后面加上nn.ReLU激活函数
            layer = nn.Sequential(layer, nn.ReLU(True))
            # 创建一个xyz_encoding_{i+1}名称的属性，并赋值为layer
            # 简单理解例如：self.xyz_encoding_1 = layer
            setattr(self, f"xyz_cls_encoding_{i+1}", layer)
    
        # output layers
        # sigma表示物体的体密度
        self.sigma = nn.Sequential(nn.Linear(self.width, self.width),
                                   nn.ReLU(True),
                                   nn.Linear(self.width, 1))

        # 输出光照信息，用3阶计算，RGB的3通道共27个参数，每通道的基函数带一组系数，共9个
        self.sh = nn.Sequential(nn.Linear(self.width, self.width),
                                   nn.ReLU(True),
                                   nn.Linear(self.width, 3 * (self.deg + 1)**2))
    # 输入光线的位置和方向信息
    def forward(self, x, clss, dirs):
        # x为空间位置信息
        input_xyz = x
        input_cls = clss
        input_xyz_cls = torch.cat([input_xyz, input_cls], 1)
        
        xyz_cls_ = input_xyz_cls

        # 对编码层进行前向传播计算
        for i in range(self.depth):
            # 如果传播层是跳跃层，将原始输入和前一层的输入合并在一起传播
            if i in self.skips:
                xyz_cls_ = torch.cat([input_xyz_cls, xyz_cls_], -1)
            # 获取每一层的layer计算
            # 简单理解，例如：xyz_ = self.xyz_encoding_1(xyz_)
            # 和正常的反向传播一样，就是写法比较高级
            xyz_cls_ = getattr(self, f"xyz_cls_encoding_{i+1}")(xyz_cls_)

        # 输出sigma层，体密度
        # [X, 1]
        sigma = self.sigma(xyz_cls_)
        # 输出光照信息
        # [X, 27]
        sh = self.sh(xyz_cls_)
        # 就对sh进行解码，输出RGB信息
        # [X, 3]
        rgb = eval_sh(deg=self.deg, sh=sh.reshape(-1, 3, (self.deg + 1)**2), dirs=dirs) # sh: [..., C, (deg + 1) ** 2]
        # rgb过sigmoid到0-1之间
        rgb = torch.sigmoid(rgb)
        # if extract_time:
        out = torch.cat([sigma, rgb, sh], -1)
        
        return out

# NERF网络
class NeRF_Rays(nn.Module):
    def __init__(self, enerf_args):
        super(NeRF_Rays, self).__init__()
        # 体素格的数据存储数量
        self.max_for_rays = 640
        # 3通道，所以乘以3
        self.in_channels_xyz = self.max_for_rays*63
        # 与位置编码同理 
        self.in_channels_cls = (2*enerf_args["emb_freqs_cls"] + 1) #13
        # 合并输入
        self.in_channels_all = self.in_channels_xyz + self.in_channels_cls
        
            
        # Nerf核心编码网络的深度，理解为MLP有多少层
        self.depth = enerf_args["fine_MLP_depth"]
        # self.depth = 50
        # 编码网络每一层的宽度，理解为MLP每一层神经元数量
        self.width = enerf_args["fine_MLP_width"]
        # self.width = 128
        # 跳跃链接层，列表形式，指定具体层数进行跳跃链接
        # self.skips = [i for i in range(5, 50, 10)]
        self.skips = enerf_args["fine_MLP_skip"]
        # 球谐函数的阶数，阶数越高，描述光照变化情况越准确
        # 一般描述变化比较平滑的环境漫反射部分，用3阶SH就足够了
        self.deg = enerf_args["fine_MLP_deg"]

        # # xyz编码网络层进行构建
        # for i in range(self.depth):
        #     # 第一层直接线性链接
        #     if i == 0:
        #         # 网络输入神经元数量是in_channels_all
        #         layer = nn.Linear(self.in_channels_all, self.width)
        #     elif i == self.skips[0]:
        #         layer = nn.Linear(self.width + self.in_channels_all, self.width)
        #     # 跳跃层输入神经元数量需要加上in_channels_xyz
        #     elif i in self.skips[1:]:
        #         # 多出in_channels_all个神经元
        #         layer = nn.Linear(self.width + self.width, self.width)
        #     else:
        #         # 其余层直接隐含层，W对W链接
        #         layer = nn.Linear(self.width, self.width)
        #     # 每一层后面加上nn.ReLU激活函数
        #     layer = nn.Sequential(layer, nn.ReLU(True))
        #     # 创建一个xyz_encoding_{i+1}名称的属性，并赋值为layer
        #     # 简单理解例如：self.xyz_encoding_1 = layer
        #     setattr(self, f"xyz_cls_encoding_{i+1}", layer)
        
        for i in range(self.depth):
            # 第一层直接线性链接
            if i == 0:
                # 网络输入神经元数量是in_channels_all
                layer = nn.Linear(self.in_channels_all, self.width)
            # 跳跃层输入神经元数量需要加上in_channels_xyz
            elif i in self.skips:
                # 多出in_channels_all个神经元
                layer = nn.Linear(self.width + self.in_channels_all, self.width)
            else:
                # 其余层直接隐含层，W对W链接
                layer = nn.Linear(self.width, self.width)
            # 每一层后面加上nn.ReLU激活函数
            layer = nn.Sequential(layer, nn.ReLU(True))
            # 创建一个xyz_encoding_{i+1}名称的属性，并赋值为layer
            # 简单理解例如：self.xyz_encoding_1 = layer
            setattr(self, f"xyz_cls_encoding_{i+1}", layer)
      
        # output layers
        # sigma表示物体的体密度
        self.rgb = nn.Sequential(nn.Linear(self.width, self.width),
                                   nn.ReLU(True),
                                   nn.Linear(self.width, 3))
        

    # 输入光线的位置和方向信息
    def forward(self, x, clss):
        # x为空间位置信息
        input_xyz = x
        input_cls = clss
        input_xyz_cls = torch.cat([input_xyz, input_cls], 1)
        
        xyz_cls_ = input_xyz_cls

        # 对编码层进行前向传播计算
        # 对编码层进行前向传播计算
        for i in range(self.depth):
            # 如果传播层是跳跃层，将原始输入和前一层的输入合并在一起传播
            if i in self.skips:
                xyz_cls_ = torch.cat([self.in_channels_all, xyz_cls_], -1)
            # 获取每一层的layer计算
            # 简单理解，例如：xyz_ = self.xyz_encoding_1(xyz_)
            # 和正常的反向传播一样，就是写法比较高级
            xyz_cls_ = getattr(self, f"xyz_cls_encoding_{i+1}")(xyz_cls_)


        # 输出rgb层，体密度
        # [Batch, 640*3]
        rgb = self.rgb(xyz_cls_)
        rgb_all_sample = torch.sigmoid(rgb)
        
        return rgb_all_sample
