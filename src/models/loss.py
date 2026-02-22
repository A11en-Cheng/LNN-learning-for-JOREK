import torch
import torch.nn as nn
from torch_geometric.utils import degree

def compute_spatial_weights(psi, batch_index, axis_weight=10.0, pedestal_weight=5.0):
    """
    计算空间权重 (Spatial Weighting) 以解决 GNN 的过平滑和磁轴振荡问题。
    Args:
        psi: 归一化前或归一化后的极向磁通 [Total_nodes, 1] 或 [Total_nodes]
        batch_index: PyG 的 batch 索引向量 [Total_nodes]
        axis_weight: 磁轴区域的权重放大倍数
        pedestal_weight: 台基区域的权重放大倍数
    Returns:
        weights: 节点空间权重 [Total_nodes, 1]
    """
    if psi.dim() == 2:
        psi = psi.squeeze(1)
        
    weights = torch.ones_like(psi)
    num_graphs = int(batch_index.max().item()) + 1
    
    for i in range(num_graphs):
        mask = (batch_index == i)
        psi_g = psi[mask]
        
        # 计算归一化磁通 psi_n = (psi - psi_axis) / (psi_edge - psi_axis)
        # 在 JOREK 中，通常磁轴处 psi 最小，边界处 psi 最大
        psi_min = psi_g.min()
        psi_max = psi_g.max()
        
        # 防止除以 0
        if psi_max - psi_min > 1e-6:
            psi_n = (psi_g - psi_min) / (psi_max - psi_min)
        else:
            psi_n = torch.zeros_like(psi_g)
            
        # 1. 磁轴区域 (Magnetic Axis): psi_n < 0.05
        axis_mask = (psi_n < 0.05)
        
        # 2. 台基区域 (Pedestal): psi_n > 0.85
        pedestal_mask = (psi_n > 0.85)
        
        # 分配权重
        graph_weights = torch.ones_like(psi_g)
        graph_weights[axis_mask] = axis_weight
        graph_weights[pedestal_mask] = pedestal_weight
        
        weights[mask] = graph_weights
        
    return weights.unsqueeze(1)

class JorekLoss(nn.Module):
    """
    针对 JOREK 破裂预测定制的复合损失函数。
    包含物理状态的回归损失、破裂概率的分类损失，以及物理诊断信息。
    """
    def __init__(self, lambda_reg=1.0, lambda_cls=0.1, lambda_phys=0.0, delta=1.0, channel_weights=None):
        super(JorekLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.lambda_phys = lambda_phys # 为未来的物理 PDE 残差预留权重
        
        # 1. 回归损失 (Regression Loss)
        # 保持 reduction='none' 以便后续应用节点权重和分离通道
        self.reg_loss = nn.HuberLoss(delta=delta, reduction='none')
        
        # 通道权重映射: [Psi, Phi, Rho, Te, v_par]
        if channel_weights is None:
            self.channel_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            self.channel_weights = torch.tensor(channel_weights)
            
        # 变量名称，用于诊断输出
        self.var_names = ["loss_psi", "loss_phi", "loss_rho", "loss_Te", "loss_vpar"]
        
        # 2. 分类损失 (Classification Loss)
        # 警告: Decoder 的 disruption_head 输出不需要加 Sigmoid！直接输出 logits。
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_states, target_states, pred_logits_disrupt, target_p_disrupt, node_weights=None):
        """
        Args:
            pred_states: 预测的物理状态 [Total_nodes, 5]
            target_states: 真实的物理状态 [Total_nodes, 5]
            pred_logits_disrupt: 预测的破裂概率 (未经过 Sigmoid) [Batch_size, 1]
            target_p_disrupt: 真实的破裂标签 (0 或 1) [Batch_size, 1]
            node_weights: (可选) 节点的空间重要性权重 [Total_nodes, 1]
        """
        device = pred_states.device
        weights = self.channel_weights.to(device)
        
        # --- 1. 计算回归损失 ---
        # raw_loss shape: [Total_nodes, 5]
        raw_loss = self.reg_loss(pred_states, target_states) 
        
        # 应用空间节点权重 (如果提供，例如针对 X 点或 Core 区域加权)
        if node_weights is not None:
            # 确保 node_weights 的形状是 [Total_nodes, 1] 以便广播
            if node_weights.dim() == 1:
                node_weights = node_weights.unsqueeze(1)
            raw_loss = raw_loss * node_weights
            
        # 应用通道权重
        weighted_loss = raw_loss * weights
        
        # 整体回归 Loss
        loss_reg = weighted_loss.mean()
        
        # 【关键改进】：提取各个通道的独立 Loss 用于诊断监控
        channel_losses = raw_loss.mean(dim=0) # Shape: [5]
        diagnostics = {
            self.var_names[i]: channel_losses[i].item() for i in range(5)
        }
        
        # --- 2. 计算分类损失 ---
        loss_cls = torch.tensor(0.0, device=device)
        if pred_logits_disrupt is not None and target_p_disrupt is not None:
            loss_cls = self.cls_loss(pred_logits_disrupt, target_p_disrupt)
            
        # --- 3. 物理软约束 (预留) ---
        loss_phys = torch.tensor(0.0, device=device)
        if self.lambda_phys > 0:
            # 未来在这里调用物理计算，例如:
            # loss_phys = compute_energy_conservation_loss(pred_states)
            pass
            
        # --- 4. 总损失组合 ---
        total_loss = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls + self.lambda_phys * loss_phys
        
        # 将整体 loss 也加入字典，方便训练循环只接收一个字典
        diagnostics.update({
            "loss_total": total_loss.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
            "loss_phys": loss_phys.item()
        })
        
        return total_loss, diagnostics