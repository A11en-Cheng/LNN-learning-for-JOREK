import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.model502_schema import (
    FIELD_INDEX,
    validate_background_source_tensor,
    validate_state_tensor,
)
from src.physics.grad_ops import compute_gradient, compute_divergence
from src.physics.mhd import get_magnetic_field
from src.physics.cocos8 import reconstruct_velocity


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


def compute_normalized_flux(psi, batch_index, psi_axis, psi_bnd, eps: float = 1e-6):
    """根据每个图的 psi_axis 与 psi_bnd 计算节点级的归一化磁通 psi_n。

    Args:
        psi: 节点极向磁通 [Total_nodes] 或 [Total_nodes, 1] (与 pred/target 中 psi 通道同单位)
        batch_index: PyG batch 索引 [Total_nodes]
        psi_axis: 每个图的磁轴磁通标量，形状 [num_graphs] 或标量
        psi_bnd: 每个图的边界磁通标量，形状 [num_graphs] 或标量
        eps: 防止除零的小常数

    Returns:
        psi_n: 归一化磁通 [Total_nodes]，理想情况下 core 区域约在 [0, 1]。
    """
    if psi.dim() == 2:
        psi = psi.squeeze(-1)

    device = psi.device
    batch_index = batch_index.to(device)

    if isinstance(psi_axis, torch.Tensor):
        psi_axis_t = psi_axis.to(device).view(-1)
    else:
        psi_axis_t = torch.tensor(psi_axis, dtype=psi.dtype, device=device).view(-1)

    if isinstance(psi_bnd, torch.Tensor):
        psi_bnd_t = psi_bnd.to(device).view(-1)
    else:
        psi_bnd_t = torch.tensor(psi_bnd, dtype=psi.dtype, device=device).view(-1)

    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    psi_n = torch.zeros_like(psi)

    for g in range(num_graphs):
        mask_g = (batch_index == g)
        if not mask_g.any():
            continue

        psi_g = psi[mask_g]

        # 防御：如果标量个数不足，则回退到该图的场最小/最大值
        if psi_axis_t.numel() > g:
            axis_val = psi_axis_t[g]
        else:
            axis_val = psi_g.min()

        if psi_bnd_t.numel() > g:
            bnd_val = psi_bnd_t[g]
        else:
            bnd_val = psi_g.max()

        denom = bnd_val - axis_val

        # 根据符号自动调整，保证 core 内 psi_n 约在 [0, 1]
        if torch.abs(denom) < eps:
            psi_n_g = torch.zeros_like(psi_g)
        elif denom > 0:
            psi_n_g = (psi_g - axis_val) / (denom + eps)
        else:
            # 若 psi 随半径减小，则反向归一化
            psi_n_g = (axis_val - psi_g) / (torch.abs(denom) + eps)

        psi_n[mask_g] = psi_n_g

    return psi_n


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
        
        # Canonical Model 502 channel weights.
        if channel_weights is None:
            cw = torch.ones(len(FIELD_INDEX), dtype=torch.float32)
        else:
            cw = torch.as_tensor(channel_weights, dtype=torch.float32)

        # 使用 register_buffer 以便在 model.to(device) 时自动迁移，并参与 state_dict 管理
        self.register_buffer("channel_weights", cw)
            
        # 变量名称，用于诊断输出
        self.var_names = [f"loss_{name}" for name in FIELD_INDEX]
        
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
        bc_natural_open=False,
        pde_residuals=None,
    ):
        """
        Args:
            pred_states: 预测的物理状态 [Total_nodes, 9]
            target_states: 真实的物理状态 [Total_nodes, 9]
            pred_logits_disrupt: 预测的破裂概率 (未经过 Sigmoid) [Batch_size, 1]
            target_p_disrupt: 真实的破裂标签 (0 或 1) [Batch_size, 1]
            node_weights: (可选) 节点的空间重要性权重 [Total_nodes, 1]
            batch_boundary: (可选) PyG Batch 拼接后的边界标记 [Total_nodes]
            freeboundary: 是否采用自由边界磁场解耦
            pde_residuals: (可选) 物理残差张量 [Total_nodes, ...]
        """
        device = pred_states.device
        validate_state_tensor(pred_states, name="JorekLoss prediction")
        validate_state_tensor(target_states, name="JorekLoss target")
        weights = self.channel_weights.to(device)
        boundary_tensor = None
        if batch_boundary is not None:
            if isinstance(batch_boundary, torch.Tensor):
                boundary_tensor = batch_boundary.to(device).view(-1)
            else:
                boundary_tensor = torch.tensor(batch_boundary, device=device).view(-1)
        internal_mask = None
        boundary_mask = None
        open_line_mask = None
        if boundary_tensor is not None:
            internal_mask = (boundary_tensor == 0)
            # boundary is a bitmask: bit2 (=4) marks open field lines / X-point legs.
            open_line_mask = (boundary_tensor & 4) != 0
            boundary_mask = boundary_tensor > 0

        if isinstance(freeboundary, torch.Tensor):
            freeboundary_flag = bool(int(freeboundary.view(-1)[0].item()))
        else:
            freeboundary_flag = bool(freeboundary)
        
        # --- 1. 计算回归损失 ---
        # raw_loss shape: [Total_nodes, 9]
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
        channel_losses = raw_loss.mean(dim=0)
        diagnostics = {
            self.var_names[i]: channel_losses[i].item() for i in range(len(self.var_names))
        }
        
        # --- 2. 计算分类损失 ---
        loss_cls = torch.tensor(0.0, device=device)
        if pred_logits_disrupt is not None and target_p_disrupt is not None:
            loss_cls = self.cls_loss(pred_logits_disrupt, target_p_disrupt)
            
        # --- 3. 内部 PDE 物理残差（带拓扑掩膜） ---
        loss_phys = torch.tensor(0.0, device=device)
        if self.lambda_phys > 0 and pde_residuals is not None and internal_mask is not None:
            # 兼容两种输入形式：
            # 1) Tensor: 单一残差场 [Total_nodes, ...] （旧行为）
            # 2) Dict: 显式给出质量与动量残差，以及计算 psi_n 所需的几何/拓扑量
            if isinstance(pde_residuals, dict):
                res_mass = pde_residuals.get("mass", None)
                res_momentum = pde_residuals.get("momentum", None)
                psi_field = pde_residuals.get("psi", None)
                psi_axis = pde_residuals.get("psi_axis", None)
                psi_bnd = pde_residuals.get("psi_bnd", None)
                batch_index = pde_residuals.get("batch", None)

                if res_mass is not None:
                    res_mass = res_mass.to(device)
                if res_momentum is not None:
                    res_momentum = res_momentum.to(device)

                # 默认：在所有内部节点上应用 PDE 约束
                mask_core = internal_mask.clone()
                mask_sol = None

                # 如果提供了 psi 及其归一化常数和 batch 信息，则构建基于磁通的 Core / SOL 区域掩膜
                if (
                    psi_field is not None
                    and psi_axis is not None
                    and psi_bnd is not None
                    and batch_index is not None
                ):
                    psi_field = psi_field.to(device)
                    batch_index = batch_index.to(device)
                    psi_n = compute_normalized_flux(psi_field, batch_index, psi_axis, psi_bnd)

                    # 1. Core Plasma (Strict Physics Region)
                    #    仅在内部节点且 psi_n <= 1.0 上施加强约束
                    mask_core = (psi_n <= 1.0) & internal_mask

                    # 2. SOL / Divertor 区域
                    mask_sol = (psi_n > 1.0)
                    if boundary_mask is not None:
                        mask_sol = mask_sol | boundary_mask

                    # 若由于数值原因 core 掩膜为空，则退化为原 internal_mask
                    if not mask_core.any():
                        mask_core = internal_mask

                # 在 Core 区域上计算质量与动量 PDE 损失；SOL 区域仅由数据与边界约束驱动
                loss_pde_mass = torch.tensor(0.0, device=device)
                loss_pde_mom = torch.tensor(0.0, device=device)

                if res_mass is not None and mask_core.any():
                    loss_pde_mass = (res_mass[mask_core] ** 2).mean()
                if res_momentum is not None and mask_core.any():
                    loss_pde_mom = (res_momentum[mask_core] ** 2).mean()

                loss_phys = loss_pde_mass + loss_pde_mom

            else:
                # 向后兼容：若仅提供一个残差张量，则在所有 internal_mask 节点上施加约束
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
                # JOREK default (mod_boundary_conditions.f90): Dirichlet on
                # psi/u/zj/w only; rho/Ti/Te/vpar/rhoimp are not Dirichlet by
                # default.  psi and zj are free on a free boundary.
                anchored = [1, 3]  # Phi(u), omega
                if not freeboundary_flag:
                    anchored = [0, 1, 2, 3]  # psi, Phi, Z_j, omega
                loss_bc = F.mse_loss(pred_bnd[:, anchored], target_bnd[:, anchored])
            
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


class JorekPhysicsLoss(nn.Module):
    """Physics-Informed Loss for interpolation / residual training.

    主要用于 HyPhy-GLNN 的物理插值 / 残差预测任务：
    - 数据项: Huber/Log-Cosh 在训练脚本中已经做了归一化，这里使用 Smooth L1 作为 data term
    - 物理项: 质量连续性 + 并行动量方程残差（在本类内部通过微分算子直接构造）
    - 边界项: 精确锚定 psi / Phi / rho / Te / v_par 的边界值

    与上面的 JorekLoss 不同，本类会自行调用 grad_ops/mhd/cocos8 来构造 PDE 残差，
    适合像 train_interpolation_test 这种“端到端物理插值”脚本。
    """

    def __init__(self, lambda_data: float = 1.0, lambda_phys: float = 0.01, lambda_bc: float = 10.0,
                 stats=None, dt: float = 1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_bc = lambda_bc
        self.dt = dt
        # stats 应为 {'mean': Tensor[9], 'std': Tensor[9]}，用于反归一化
        self.stats = stats

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """将 Z-score + symlog 映射后的特征恢复到物理空间。

        1. 先做 Z-score 反变换: x_linear = x * std + mean
        2. 再对 rho/T_i/rho_n/T_e 反 symlog: sign(y)*expm1(|y|)，并做轻微 clamp，
           避免在 PDE 残差计算中除零。
        """
        validate_state_tensor(x, name="physics-loss state")
        if self.stats is None:
            return x

        mean = self.stats['mean'].to(x.device)
        std = self.stats['std'].to(x.device)

        # 1. Reverse Z-score
        x_linear = x * std + mean

        # 2. Reverse Log-Space for Thermodynamic variables（无原地操作版本）
        psi = x_linear[:, 0:1]
        Phi = x_linear[:, 1:2]
        z_j = x_linear[:, 2:3]
        omega = x_linear[:, 3:4]

        def _symexp(x):
            return torch.sign(x) * torch.expm1(x.abs())

        rho = torch.clamp(_symexp(x_linear[:, 4:5]), min=1e-8)
        t_i = torch.clamp(_symexp(x_linear[:, 5:6]), min=1e-8)

        v_par = x_linear[:, 6:7]

        rho_n = torch.clamp(_symexp(x_linear[:, 7:8]), min=1e-8)
        t_e = torch.clamp(_symexp(x_linear[:, 8:9]), min=1e-8)

        x_phys = torch.cat([psi, Phi, z_j, omega, rho, t_i, v_par, rho_n, t_e], dim=-1)
        return x_phys

    def forward(
        self,
        pred,
        target,
        graph_data,
        curr_state=None,
        boundary=None,
        freeboundary=None,
        dt=None,
    ):
        """计算总损失（数据 + 物理 + 边界）。

        Args:
            pred: 预测的下一步节点特征 [N, 9]（归一化空间）
            target: 目标下一步节点特征 [N, 9]（归一化空间）
            graph_data: PyG Data，包含 edge_index/edge_attr/pos/batch 等
            curr_state: 当前时刻节点特征 [N, 9]（归一化），缺省为 graph_data.x
            boundary: 可选边界标记张量 [N]
            freeboundary: 自由边界 vs 固定边界 标志
            dt: 覆盖构造时的物理时间步（秒）。用于非均匀时间步训练。
        """
        device = pred.device
        step_dt = self.dt if dt is None else dt

        # 边界元数据
        boundary_tensor = None
        if boundary is None and hasattr(graph_data, 'boundary'):
            boundary = graph_data.boundary
        if boundary is not None:
            if isinstance(boundary, torch.Tensor):
                boundary_tensor = boundary.to(device).view(-1)
            else:
                boundary_tensor = torch.tensor(boundary, dtype=torch.long, device=device).view(-1)
        internal_mask = None
        boundary_mask = None
        if boundary_tensor is not None:
            internal_mask = (boundary_tensor == 0)
            boundary_mask = (boundary_tensor > 0)

        # 自由边界标志
        if freeboundary is None and hasattr(graph_data, 'freeboundary'):
            freeboundary = graph_data.freeboundary
        if isinstance(freeboundary, torch.Tensor):
            freeboundary_flag = bool(int(freeboundary.view(-1)[0].item()))
        elif freeboundary is None:
            freeboundary_flag = True
        else:
            freeboundary_flag = bool(freeboundary)

        # 1. 纯数据项（Huber/Smooth L1）
        data_loss = F.smooth_l1_loss(pred, target, reduction='mean')

        # 无物理项时，只保留数据 + 边界约束，避免多余的计算
        if self.lambda_phys <= 0.0:
            loss_bc = torch.tensor(0.0, device=device)
            if boundary_mask is not None and boundary_mask.any():
                pred_bnd = pred[boundary_mask]
                target_bnd = target[boundary_mask]
                if pred_bnd.shape[0] > 0:
                    loss_bc_plasma = F.mse_loss(pred_bnd[:, 1:], target_bnd[:, 1:])
                    psi_penalty = 5.0 if freeboundary_flag else 50.0
                    loss_bc_psi = F.mse_loss(pred_bnd[:, 0], target_bnd[:, 0]) * psi_penalty
                    loss_bc = loss_bc_plasma + loss_bc_psi
            return data_loss + self.lambda_bc * loss_bc

        # 2. 物理 PDE 残差构造
        if curr_state is None:
            curr_state = graph_data.x

        # 反归一化当前与下一步状态
        curr_denorm = self.denormalize(curr_state)
        next_denorm = self.denormalize(pred)

        # Canonical Model 502 slices.  This remains a reduced diagnostic
        # residual, not a claim that all nine PDEs are implemented.
        rho_curr = curr_denorm[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]
        v_par_curr = curr_denorm[:, FIELD_INDEX["v_par"]:FIELD_INDEX["v_par"] + 1]

        psi_next = next_denorm[:, FIELD_INDEX["psi"]:FIELD_INDEX["psi"] + 1]
        Phi_next = next_denorm[:, FIELD_INDEX["Phi"]:FIELD_INDEX["Phi"] + 1]
        rho_next = next_denorm[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]
        t_i_next = next_denorm[:, FIELD_INDEX["T_i"]:FIELD_INDEX["T_i"] + 1]
        t_e_next = next_denorm[:, FIELD_INDEX["T_e"]:FIELD_INDEX["T_e"] + 1]
        v_par_next = next_denorm[:, FIELD_INDEX["v_par"]:FIELD_INDEX["v_par"] + 1]

        R = graph_data.pos[:, 0:1]

        # 归一化常数 F0（如有）
        F0 = 1.0
        if hasattr(graph_data, 'norm_factors') and graph_data.norm_factors is not None:
            if isinstance(graph_data.norm_factors, dict) and 'F0' in graph_data.norm_factors:
                F0 = graph_data.norm_factors['F0']
            elif hasattr(graph_data.norm_factors, 'F0'):
                F0 = graph_data.norm_factors.F0

        if isinstance(F0, torch.Tensor):
            if hasattr(graph_data, 'batch') and graph_data.batch is not None:
                F0 = F0[graph_data.batch].view(-1, 1)
            else:
                F0 = F0[0].item() if F0.numel() > 0 else 1.0

        # 梯度算子（在物理空间上）
        grad_rho = compute_gradient(rho_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_v_par = compute_gradient(v_par_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_Phi = compute_gradient(Phi_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_psi = compute_gradient(psi_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)

        # B 场与速度场重建
        B_field = get_magnetic_field(grad_psi, R, F0)  # [N, 3]
        v_field = reconstruct_velocity(grad_Phi, v_par_next, B_field, R)  # [N, 3]

        v_R = v_field[:, 0:1]
        v_Z = v_field[:, 1:2]

        # --- 质量守恒残差 ---
        rho_v_R = rho_next * v_R
        rho_v_Z = rho_next * v_Z
        rho_v_pol = torch.cat([rho_v_R, rho_v_Z], dim=-1)

        div_cart_rho_v = compute_divergence(rho_v_pol, graph_data.edge_index, graph_data.edge_attr).unsqueeze(-1)
        div_rho_v = div_cart_rho_v + rho_v_R / (R + 1e-8)

        # --- Mass Continuity Residual (Dimensionless Relative Error) ---
        source_rho = torch.zeros_like(rho_next)
        if hasattr(graph_data, "background_source") and graph_data.background_source is not None:
            validate_background_source_tensor(
                graph_data.background_source, name="physics-loss background source"
            )
            source_rho = graph_data.background_source[:, 0:1].to(
                device=device, dtype=rho_next.dtype
            )
        res_mass_raw = (rho_next - rho_curr) / step_dt + div_rho_v - source_rho
        
        # 乘以 dt 抵消时间量纲，除以 rho 抵消密度量纲。结果为纯小数 (Δρ/ρ)
        res_mass = (res_mass_raw * step_dt) / (torch.abs(rho_curr) + 1e-3)
        
        # --- Parallel Momentum Residual (Dimensionless Relative Error) ---
        v_dot_grad_v_par = v_R * grad_v_par[:, 0:1] + v_Z * grad_v_par[:, 1:2]
        # The full Model 502 pressure also depends on impurity charge-state
        # coefficients.  Those metadata are not yet part of the HDF5 contract,
        # so this is explicitly the background two-temperature approximation.
        pressure_next = rho_next * (t_i_next + t_e_next)
        grad_pressure = compute_gradient(pressure_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        
        B_R = B_field[:, 0:1]
        B_Z = B_field[:, 1:2]
        B_mag = torch.norm(B_field, p=2, dim=-1, keepdim=True) + 1e-8
        
        B_dot_grad_pressure = B_R * grad_pressure[:, 0:1] + B_Z * grad_pressure[:, 1:2]
        grad_par_pressure = B_dot_grad_pressure / B_mag
        
        res_momentum_raw = rho_next * ((v_par_next - v_par_curr) / step_dt + v_dot_grad_v_par) + grad_par_pressure
        
        # 构造参考速度：真实速度的绝对值 + 0.1 的保护底噪 (防止过零爆炸)
        v_ref = torch.abs(v_par_curr) + 0.1
        
        # 乘以 dt 抵消时间，除以 (rho * v_ref) 抵消动量密度量纲。结果为纯小数 (ΔP/P)
        res_momentum = (res_momentum_raw * step_dt) / (torch.abs(rho_curr) * v_ref + 1e-3)

        # Core/SOL 拓扑掩膜
        zero = torch.tensor(0.0, device=device)
        if internal_mask is not None and internal_mask.any():
            mask_core = internal_mask.clone()

            psi_axis = getattr(graph_data, "psi_axis", None)
            psi_bnd = getattr(graph_data, "psi_bnd", None)
            batch_index = getattr(graph_data, "batch", None)

            # 如提供 psi_axis/psi_bnd/batch，则用归一化磁通来细化 Core 区域
            if psi_axis is not None and psi_bnd is not None and batch_index is not None:
                psi_field = psi_next  # [N, 1]
                psi_n = compute_normalized_flux(psi_field, batch_index, psi_axis, psi_bnd)

                mask_core = (psi_n <= 1.0) & internal_mask

                # 若数值原因导致 core 掩膜为空，则退化为原 internal_mask
                if not mask_core.any():
                    mask_core = internal_mask

            # 在 Core 区域上施加 PDE 约束
            mass_term = F.smooth_l1_loss(
                res_mass[mask_core], torch.zeros_like(res_mass[mask_core]), beta=0.1
            )
            mom_term = F.smooth_l1_loss(
                res_momentum[mask_core], torch.zeros_like(res_momentum[mask_core]), beta=0.1
            )
        else:
            # 无可靠内部掩膜时，在所有节点上施加 PDE 约束
            mass_term = F.smooth_l1_loss(res_mass, torch.zeros_like(res_mass), beta=0.1)
            mom_term = F.smooth_l1_loss(res_momentum, torch.zeros_like(res_momentum), beta=0.1)

        phys_loss = mass_term + mom_term

        # 边界条件锚定
        loss_bc = zero
        if boundary_mask is not None and boundary_mask.any():
            pred_bnd = pred[boundary_mask]
            target_bnd = target[boundary_mask]
            if pred_bnd.shape[0] > 0:
                loss_bc_plasma = F.mse_loss(pred_bnd[:, 1:], target_bnd[:, 1:])
                psi_penalty = 5.0 if freeboundary_flag else 50.0
                loss_bc_psi = F.mse_loss(pred_bnd[:, 0], target_bnd[:, 0]) * psi_penalty
                loss_bc = loss_bc_plasma + loss_bc_psi

        total_loss = self.lambda_data * data_loss + self.lambda_phys * phys_loss + self.lambda_bc * loss_bc

        return total_loss
