import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, lambda_reg=1.0, lambda_cls=0.1, lambda_phys=0.0, lambda_bc=10.0, delta=1.0, channel_weights=None):
        super(JorekLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.lambda_phys = lambda_phys # 为未来的物理 PDE 残差预留权重
        self.lambda_bc = lambda_bc
        
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

    def forward(
        self,
        pred_states,
        target_states,
        pred_logits_disrupt,
        target_p_disrupt,
        node_weights=None,
        batch_boundary=None,
        freeboundary=True,
        pde_residuals=None,
    ):
        """
        Args:
            pred_states: 预测的物理状态 [Total_nodes, 5]
            target_states: 真实的物理状态 [Total_nodes, 5]
            pred_logits_disrupt: 预测的破裂概率 (未经过 Sigmoid) [Batch_size, 1]
            target_p_disrupt: 真实的破裂标签 (0 或 1) [Batch_size, 1]
            node_weights: (可选) 节点的空间重要性权重 [Total_nodes, 1]
            batch_boundary: (可选) PyG Batch 拼接后的边界标记 [Total_nodes]
            freeboundary: 是否采用自由边界磁场解耦
            pde_residuals: (可选) 物理残差张量 [Total_nodes, ...]
        """
        device = pred_states.device
        weights = self.channel_weights.to(device)
        boundary_tensor = None
        if batch_boundary is not None:
            if isinstance(batch_boundary, torch.Tensor):
                boundary_tensor = batch_boundary.to(device).view(-1)
            else:
                boundary_tensor = torch.tensor(batch_boundary, device=device).view(-1)
        internal_mask = None
        boundary_mask = None
        if boundary_tensor is not None:
            internal_mask = (boundary_tensor == 0)
            boundary_mask = (boundary_tensor > 0)

        if isinstance(freeboundary, torch.Tensor):
            freeboundary_flag = bool(int(freeboundary.view(-1)[0].item()))
        else:
            freeboundary_flag = bool(freeboundary)
        
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
            
        # --- 3. 内部 PDE 物理残差 ---
        loss_phys = torch.tensor(0.0, device=device)
        if self.lambda_phys > 0 and pde_residuals is not None and internal_mask is not None:
            if isinstance(pde_residuals, torch.Tensor):
                residuals = pde_residuals.to(device)
            else:
                residuals = torch.tensor(pde_residuals, dtype=pred_states.dtype, device=device)
            if internal_mask.any():
                loss_phys = (residuals[internal_mask] ** 2).mean()

        # --- 4. 精确边界条件约束 ---
        loss_bc = torch.tensor(0.0, device=device)
        if boundary_mask is not None and boundary_mask.any():
            pred_bnd = pred_states[boundary_mask]
            target_bnd = target_states[boundary_mask]
            if pred_bnd.shape[0] > 0:
                loss_bc_plasma = F.mse_loss(pred_bnd[:, 1:], target_bnd[:, 1:])
                psi_penalty = 5.0 if freeboundary_flag else 50.0
                loss_bc_psi = F.mse_loss(pred_bnd[:, 0], target_bnd[:, 0]) * psi_penalty
                loss_bc = loss_bc_plasma + loss_bc_psi
            
        # --- 5. 总损失组合 ---
        total_loss = (
            self.lambda_reg * loss_reg
            + self.lambda_cls * loss_cls
            + self.lambda_phys * loss_phys
            + self.lambda_bc * loss_bc
        )
        
        # 将整体 loss 也加入字典，方便训练循环只接收一个字典
        diagnostics.update({
            "loss_total": total_loss.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
            "loss_phys": loss_phys.item(),
            "loss_bc": loss_bc.item()
        })
        
        return total_loss, diagnostics