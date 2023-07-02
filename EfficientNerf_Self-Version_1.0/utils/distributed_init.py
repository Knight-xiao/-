import os
import torch
import logging

# 分布式步骤一:
# 初始化进程组，同时初始化 distributed
# 包，然后才能使用 distributed 包的其他函数。
class Distributed_config():
    def __init__(self, sys_param):
        self.init_distributed_mode(sys_param)
  
    # 初始化多卡训练模式
    def init_distributed_mode(self, sys_param):
        # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
        # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU，RANK和LOCAL_RANK相等
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # 这俩表示当前的进程数id，由0-world_size，指定几块卡，就会有几个进程
            sys_param['rank'] = int(os.environ["RANK"])
            sys_param['gpu'] = int(os.environ['LOCAL_RANK'])
            # world_size由terminal终端的--nproc_per_node指定
            sys_param['world_size'] = int(os.environ['WORLD_SIZE'])

        else:
            logging.info('多卡训练关闭')
            sys_param['distributed'] = False
            return

        # 分布式的Flag变为True
        start_device = sys_param["start_device"]
        sys_param['gpu'] = sys_param['gpu'] + start_device

        logging.info('多卡训练开启:分布式训练当前线程{},使用GPU:{}'.format(sys_param['rank'], sys_param['gpu']))
        sys_param['distributed'] = True

        # 初始化相应的gpu
        torch.cuda.set_device(sys_param['gpu'])

        # 后端nccl：Nvidia Collective multi-GPU Communication Library
        # 多卡通信的一种方式
        sys_param['dist_backend'] = 'nccl'

        # backend str/Backend 是通信所用的后端
        # init_method str 这个URL指定了如何初始化互相通信的进程
        # world_size int 执行训练的所有的进程数，和卡的数量一致最好
        # rank int this进程的编号，也是其优先级
        torch.distributed.init_process_group(backend=sys_param['dist_backend'], 
                                             world_size=sys_param['world_size'], 
                                             rank=sys_param['rank'])

        # pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，
        # 然后其它进程从缓存中读取，不同进程之间的数据同步具体通过torch.distributed.barrier()实现
        torch.distributed.barrier(device_ids=[sys_param['gpu']])

        self.setup_for_distributed(sys_param['rank'] == 0)

    # 如果进程不是主进程，就不打印信息
    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print
        logging_info = logging.info
        
        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        
        def info(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                logging_info(*args, **kwargs)            
     
        __builtin__.print = print
        logging.info = info