import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from torch_geometric.nn import GATConv, LayerNorm, global_mean_pool
from src.data.model502_schema import NUM_STATE_FIELDS, validate_state_tensor


def _silu_block(conv, norm, x, edge_index, edge_attr):
    return F.silu(norm(conv(x, edge_index, edge_attr=edge_attr)))

class JorekGNNDecoder(nn.Module):
    """
    Reconstruction (GNN Decoder) - Split Architecture:
    将 LNN 处理后的潜在状态解码回网格节点，预测 Model 502 的 9 场残差。
    采用双头架构 (Split Decoder)：
    - Head 1 (smooth/electromagnetic): psi, Phi, Z_j, omega, v_parallel
    - Head 2 (thermodynamic/impurity): rho, T_i, rho_imp, T_e
    """
    def __init__(
        self,
        latent_dim=32,
        hidden_dim=64,
        output_dim=NUM_STATE_FIELDS,
        num_layers=2,
        edge_dim=2,
        checkpoint=False,
    ):
        super(JorekGNNDecoder, self).__init__()
        if output_dim != NUM_STATE_FIELDS:
            raise ValueError(
                f"Model 502 decoder requires output_dim={NUM_STATE_FIELDS}, got {output_dim}. "
                "Use an explicit legacy adapter for five-field checkpoints."
            )
        
        self.num_layers = num_layers
        self.checkpoint = bool(checkpoint)
        
        # GATConv 参数
        heads = 2
        concat = True
        
        # ==========================================
        # 共享特征提取层 (Shared Base)
        # ==========================================
        self.shared_conv = GATConv(latent_dim, hidden_dim // heads, heads=heads, concat=concat, edge_dim=edge_dim)
        self.shared_norm = LayerNorm(hidden_dim)
        
        # ==========================================
        # Head 1: smooth/electromagnetic fields: psi, Phi, Z_j, omega, v_par
        # ==========================================
        self.smooth_convs = nn.ModuleList()
        self.smooth_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.smooth_convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=concat, edge_dim=edge_dim))
            self.smooth_norms.append(LayerNorm(hidden_dim))
        self.smooth_head = nn.Linear(hidden_dim, 5)
        
        # ==========================================
        # Head 2: thermodynamic/impurity fields: rho, T_i, rho_imp, T_e
        # 使用 GATConv 增强对局部梯度的感知能力，并增加隐藏层维度
        # ==========================================
        thermo_hidden = hidden_dim * 2
        self.thermo_proj = nn.Linear(hidden_dim, thermo_hidden)
        self.thermo_convs = nn.ModuleList()
        self.thermo_norms = nn.ModuleList()
        # 增加一层深度
        for _ in range(num_layers):
            # 使用 GATConv, heads=2
            self.thermo_convs.append(GATConv(thermo_hidden, thermo_hidden // heads, heads=heads, concat=concat, edge_dim=edge_dim))
            self.thermo_norms.append(LayerNorm(thermo_hidden))
        self.thermo_head = nn.Linear(thermo_hidden, 4)
        
        # ==========================================
        # 破裂预测头 (Disruption Head)
        # ==========================================
        self.disruption_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, latent_x, edge_index, batch_index, pos):
        """
        Args:
            latent_x: LNN 输出的潜在特征 [Total_nodes, latent_dim]
            edge_index: 图连接矩阵 [2, Num_edges]
            batch_index: PyG 的 batch 索引向量 [Total_nodes]
            pos: 节点坐标矩阵 [Total_nodes, 2] (R, Z)
        Returns:
            state_vars: 归一化空间中的单步状态残差 [Total_nodes, 9]
            p_disrupt: 破裂概率 [Batch_size, 1]
        """
        # 计算几何特征 (Delta R, Delta Z) 作为 edge_attr
        row, col = edge_index
        edge_attr = pos[row] - pos[col] # [Num_edges, 2]
        
        # 1. 共享特征提取
        use_checkpoint = self.checkpoint and self.training
        if use_checkpoint:
            x_shared = _torch_checkpoint(
                _silu_block,
                self.shared_conv, self.shared_norm, latent_x, edge_index, edge_attr,
                use_reentrant=False,
            )
        else:
            x_shared = _silu_block(
                self.shared_conv, self.shared_norm, latent_x, edge_index, edge_attr
            )
        
        # 2. Head 1: Smooth Fields (psi, Phi, v_par)
        x_smooth = x_shared
        for i in range(self.num_layers - 1):
            if use_checkpoint:
                x_smooth = _torch_checkpoint(
                    _silu_block,
                    self.smooth_convs[i], self.smooth_norms[i], x_smooth, edge_index, edge_attr,
                    use_reentrant=False,
                )
            else:
                x_smooth = _silu_block(
                    self.smooth_convs[i], self.smooth_norms[i], x_smooth, edge_index, edge_attr
                )
        out_smooth = self.smooth_head(x_smooth) # [Total_nodes, 5]
        
        # 3. Head 2: Thermodynamic Fields (rho, T)
        x_thermo = self.thermo_proj(x_shared)
        x_thermo = F.silu(x_thermo)
        for i in range(self.num_layers):
            if use_checkpoint:
                x_thermo = _torch_checkpoint(
                    _silu_block,
                    self.thermo_convs[i], self.thermo_norms[i], x_thermo, edge_index, edge_attr,
                    use_reentrant=False,
                )
            else:
                x_thermo = _silu_block(
                    self.thermo_convs[i], self.thermo_norms[i], x_thermo, edge_index, edge_attr
                )
        out_thermo = self.thermo_head(x_thermo) # [Total_nodes, 4]
        
        # 强制物理正定性 (Hard Constraint: Positivity for rho and T)
        # 预测 log(rho) 和 log(T) 是一种方法，但为了与现有 Loss 兼容，
        # 我们继续使用 Softplus，但由于 GATConv 的引入，拟合能力会增强。
        # 注意：由于全局归一化 (Z-score) 的存在，网络输出的实际上是归一化后的值，
        # 归一化后的值是可以为负的。因此，在网络输出端强制 Softplus 可能会导致
        # 无法预测出低于全局均值的 rho 和 T。
        # 真正的物理正定性应该在反归一化之后保证。
        # 为了修复这个问题，我们移除这里的 Softplus，让网络自由输出 Z-score。
        # out_thermo = torch.nn.functional.softplus(out_thermo)
        
        # 4. Reassemble the canonical Model 502 order:
        # [psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e].
        state_vars = torch.empty((latent_x.size(0), NUM_STATE_FIELDS), device=latent_x.device)
        state_vars[:, 0] = out_smooth[:, 0] # psi
        state_vars[:, 1] = out_smooth[:, 1] # Phi
        state_vars[:, 2] = out_smooth[:, 2] # Z_j
        state_vars[:, 3] = out_smooth[:, 3] # omega
        state_vars[:, 4] = out_thermo[:, 0] # rho
        state_vars[:, 5] = out_thermo[:, 1] # T_i
        state_vars[:, 6] = out_smooth[:, 4] # v_par
        state_vars[:, 7] = out_thermo[:, 2] # rho_imp
        state_vars[:, 8] = out_thermo[:, 3] # T_e
        validate_state_tensor(state_vars, name="decoder residual")
        
        # 5. 预测图级别的破裂概率
        pooled_x = global_mean_pool(x_shared, batch_index) # [Batch_size, hidden_dim]
        p_disrupt = self.disruption_head(pooled_x)  # [Batch_size, 1]
        
        return state_vars, p_disrupt
