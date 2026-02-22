import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, LayerNorm, global_mean_pool

class JorekGNNDecoder(nn.Module):
    """
    Reconstruction (GNN Decoder) - Split Architecture:
    将 LNN 处理后的潜在状态解码回网格节点，预测 5 个核心物理量。
    采用双头架构 (Split Decoder)：
    - Head 1 (Smooth Fields): 预测 psi, Phi, v_parallel (使用 GATConv)
    - Head 2 (Thermodynamic Fields): 预测 rho, T (使用更深的 GATConv，以拟合陡峭的台基梯度)
    """
    def __init__(self, latent_dim=32, hidden_dim=64, output_dim=5, num_layers=2, edge_dim=2):
        super(JorekGNNDecoder, self).__init__()
        
        self.num_layers = num_layers
        
        # GATConv 参数
        heads = 2
        concat = True
        
        # ==========================================
        # 共享特征提取层 (Shared Base)
        # ==========================================
        self.shared_conv = GATConv(latent_dim, hidden_dim // heads, heads=heads, concat=concat, edge_dim=edge_dim)
        self.shared_norm = LayerNorm(hidden_dim)
        
        # ==========================================
        # Head 1: Smooth Fields (psi, Phi, v_par)
        # 索引: 0, 1, 4
        # ==========================================
        self.smooth_convs = nn.ModuleList()
        self.smooth_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.smooth_convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=concat, edge_dim=edge_dim))
            self.smooth_norms.append(LayerNorm(hidden_dim))
        self.smooth_head = nn.Linear(hidden_dim, 3) # 输出 3 个变量
        
        # ==========================================
        # Head 2: Thermodynamic Fields (rho, T)
        # 索引: 2, 3
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
        self.thermo_head = nn.Linear(thermo_hidden, 2) # 输出 2 个变量
        
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
            state_vars: 预测的物理状态 [Total_nodes, 5]
            p_disrupt: 破裂概率 [Batch_size, 1]
        """
        # 计算几何特征 (Delta R, Delta Z) 作为 edge_attr
        row, col = edge_index
        edge_attr = pos[row] - pos[col] # [Num_edges, 2]
        
        # 1. 共享特征提取
        x_shared = self.shared_conv(latent_x, edge_index, edge_attr=edge_attr)
        x_shared = self.shared_norm(x_shared)
        x_shared = torch.nn.functional.silu(x_shared)
        
        # 2. Head 1: Smooth Fields (psi, Phi, v_par)
        x_smooth = x_shared
        for i in range(self.num_layers - 1):
            x_smooth = self.smooth_convs[i](x_smooth, edge_index, edge_attr=edge_attr)
            x_smooth = self.smooth_norms[i](x_smooth)
            x_smooth = torch.nn.functional.silu(x_smooth)
        out_smooth = self.smooth_head(x_smooth) # [Total_nodes, 3]
        
        # 3. Head 2: Thermodynamic Fields (rho, T)
        x_thermo = self.thermo_proj(x_shared)
        x_thermo = torch.nn.functional.silu(x_thermo)
        for i in range(self.num_layers):
            x_thermo = self.thermo_convs[i](x_thermo, edge_index, edge_attr=edge_attr)
            x_thermo = self.thermo_norms[i](x_thermo)
            x_thermo = torch.nn.functional.silu(x_thermo)
        out_thermo = self.thermo_head(x_thermo) # [Total_nodes, 2]
        
        # 强制物理正定性 (Hard Constraint: Positivity for rho and T)
        # 预测 log(rho) 和 log(T) 是一种方法，但为了与现有 Loss 兼容，
        # 我们继续使用 Softplus，但由于 GATConv 的引入，拟合能力会增强。
        # 注意：由于全局归一化 (Z-score) 的存在，网络输出的实际上是归一化后的值，
        # 归一化后的值是可以为负的。因此，在网络输出端强制 Softplus 可能会导致
        # 无法预测出低于全局均值的 rho 和 T。
        # 真正的物理正定性应该在反归一化之后保证。
        # 为了修复这个问题，我们移除这里的 Softplus，让网络自由输出 Z-score。
        # out_thermo = torch.nn.functional.softplus(out_thermo)
        
        # 4. 拼接输出 (重组为 [psi, Phi, rho, T, v_par])
        state_vars = torch.empty((latent_x.size(0), 5), device=latent_x.device)
        state_vars[:, 0] = out_smooth[:, 0] # psi
        state_vars[:, 1] = out_smooth[:, 1] # Phi
        state_vars[:, 2] = out_thermo[:, 0] # rho
        state_vars[:, 3] = out_thermo[:, 1] # T
        state_vars[:, 4] = out_smooth[:, 2] # v_par
        
        # 5. 预测图级别的破裂概率
        pooled_x = global_mean_pool(x_shared, batch_index) # [Batch_size, hidden_dim]
        p_disrupt = self.disruption_head(pooled_x)  # [Batch_size, 1]
        
        return state_vars, p_disrupt
