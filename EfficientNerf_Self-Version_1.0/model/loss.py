import torch
import torch.nn as nn
import logging

class Efficient_NeEF_Loss(nn.Module):
    def __init__(self, sys_param):
        super(Efficient_NeEF_Loss, self).__init__()
        self.sys_param = sys_param
        # 损失函数类型，默认 self.loss_type
        self.loss_type = self.sys_param['loss_type']
        # 均值损失
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss(reduction='mean')
        # 其他策略
        else:
            pass
    # 损失计算
    def forward(self, inputs, targets):
        loss = 0.0
        # distill
        if 'rgb_rays' in inputs:
            loss_model = self.loss(inputs['rgb_fine'], inputs['rgb_rays'])
            loss_gt   = self.loss(inputs['rgb_rays'], targets)
            # if loss_mode > loss_gt:
            #     logging.info("提纯中...")
            #     loss = loss_gt
            # else:
            #     loss = loss_mode + loss_gt

            return loss_model + loss_gt

        if 'rgb_coarse' in inputs:
            loss += self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss