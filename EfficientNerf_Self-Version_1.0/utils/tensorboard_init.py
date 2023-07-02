import os
import shutil 

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Tensorboard_config():
    def __init__(self, sys_param):
        if sys_param['tb_available']:
            # tensorboard生成文件的保存路径
            self.save_pth = os.path.join(Path("results"), Path(sys_param["tb_pth"]))
            # 主进程操作
            if (sys_param["distributed"]) and (sys_param['rank'] == 0):
                # 如果是删除模式，就先删除文件夹和所有数据，再重新创建
                if (sys_param['tb_del']):
                    if not os.path.exists(self.save_pth):
                        os.makedirs(self.save_pth)
                    else:
                        shutil.rmtree(self.save_pth)
                        os.makedirs(self.save_pth)
                # 如果不是，就直接检验路径
                else:
                    if not os.path.exists(self.save_pth):
                        os.makedirs(self.save_pth)
            # 初始化
            self.tblogger = SummaryWriter(self.save_pth)
        else:
            # 初始化
            self.tblogger = None
        
    def add_scalar(self, *args, **kwargs):
        # 添加变量
        self.tblogger.add_scalar(*args, **kwargs)