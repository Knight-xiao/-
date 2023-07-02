import logging
import time
import os

from pathlib import Path

class Log_config():
    def __init__(self, sys_param):
        # 读取保存状态
        self.save_mode = sys_param['log']
        self.save_pth = sys_param['log_pth']
        self.log_function_start()

    # 判断是否要生成日志，如果需要生成日志，就读取存储路径，如果不生成，就直接设置显示模式
    # 如果生成日志，则不显示日志
    def log_function_start(self):
        if self.save_mode:
            # 读取日志文件的存储路径
            #判断存储路径是否存在，如果不存在就创造一个路径
            results_log_pth = os.path.join(Path("results"), Path(self.save_pth))
            if not os.path.exists(results_log_pth):
                os.makedirs(results_log_pth)

            # 获取当时的时间，作为log的名字
            ticks = time.asctime(time.localtime(time.time()) )
            ticks = str(ticks).replace(' ', '-').replace(':','-')
            log_name = '{}.log'.format(os.path.join(self.save_pth, ticks))

            logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                                datefmt='%m/%d/%Y %H:%M:%S', 
                                level=logging.INFO,
                                filemode='a',
                                filename=log_name)
        else:      
            # 设置代码运行过程中的log信息，在代码调试过程中大家往往用print验证输出，但是大型代码往往需要
            # 记录很多的节点信息，往往这些信息是存入文件供人查看的，实现这个功能的方法就是使用logging模块
            # 在打印输出信息的时候，log完全可以代替print
            # 设置log的输出格式
            logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                                datefmt='%m/%d/%Y %H:%M:%S', 
                                level=logging.INFO)