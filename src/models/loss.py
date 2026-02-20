import torch
import torch.nn as nn

class JorekLoss(nn.Module):
    """
    针对 JOREK 破裂预测定制的复合损失函数。
    包含物理状态的回归损失和破裂概率的分类损失。
    """
    def __init__(self, lambda_reg=1.0, lambda_cls=0.1, delta=1.0):
        super(JorekLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        
        # 1. 回归损失 (Regression Loss)
        # 使用 Huber Loss (Smooth L1 Loss) 代替 MSE。
        # 当误差较小时表现为 L2 loss，误差较大时表现为 L1 loss。
        # 这对于处理破裂期间物理量指数级增长导致的异常值非常关键，能防止梯度爆炸。
        self.reg_loss = nn.HuberLoss(delta=delta)
        
        # 2. 分类损失 (Classification Loss)
        # 用于预测破裂概率 P_disruption。
        # 因为 Decoder 的 disruption_head 已经包含了 Sigmoid，所以这里使用 BCELoss。
        self.cls_loss = nn.BCELoss()

    def forward(self, pred_states, target_states, pred_p_disrupt, target_p_disrupt):
        """
        Args:
            pred_states: 预测的物理状态 [Total_nodes, 5]
            target_states: 真实的物理状态 (下一时间步) [Total_nodes, 5]
            pred_p_disrupt: 预测的破裂概率 [Batch_size, 1]
            target_p_disrupt: 真实的破裂标签 (0 或 1) [Batch_size, 1]
        """
        # 计算回归损失
        loss_reg = self.reg_loss(pred_states, target_states)
        
        # 计算分类损失
        # BCELoss requires inputs in range [0, 1]. To avoid numerical issues with exact 0 or 1, we clamp the input.
        loss_cls = self.cls_loss(torch.clamp(pred_p_disrupt, 1e-7, 1 - 1e-7), target_p_disrupt)
        
        # 组合损失
        total_loss = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls
        
        return total_loss, loss_reg, loss_cls
