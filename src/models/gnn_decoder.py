import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, LayerNorm, global_mean_pool

class JorekGNNDecoder(nn.Module):
    """
    Reconstruction (GNN Decoder):
    将 LNN 处理后的潜在状态解码回网格节点，预测 5 个核心物理量。
    同时包含一个辅助的分类头，用于预测破裂概率 (P_disruption)。
    """
    def __init__(self, latent_dim=32, hidden_dim=64, output_dim=5, num_layers=2):
        super(JorekGNNDecoder, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 解码层
        self.convs.append(SAGEConv(latent_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
            
        # 1. 物理状态预测头 (State Head)
        # 预测 5 个核心变量: psi, Phi, rho, T, v_parallel
        self.state_head = nn.Linear(hidden_dim, output_dim)
        
        # 2. 破裂预测头 (Disruption Head)
        # 辅助分类头，预测破裂概率 P_disruption
        self.disruption_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出概率 [0, 1]
        )

    def forward(self, latent_x, edge_index, batch_index):
        """
        Args:
            latent_x: LNN 输出的潜在特征 [Total_nodes, latent_dim]
            edge_index: 图连接矩阵 [2, Num_edges]
            batch_index: PyG 的 batch 索引向量 [Total_nodes]
        Returns:
            state_vars: 预测的物理状态 [Total_nodes, 5]
            p_disrupt: 破裂概率 [Batch_size, 1]
        """
        x = latent_x
        
        # GNN 解码
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = torch.nn.functional.silu(x)
            
        # 1. 预测节点级别的物理状态
        state_vars = self.state_head(x) # [Total_nodes, 5]
        
        # 2. 预测图级别的破裂概率
        # 使用 global_mean_pool 将节点特征聚合为图特征
        pooled_x = global_mean_pool(x, batch_index) # [Batch_size, hidden_dim]
        p_disrupt = self.disruption_head(pooled_x)  # [Batch_size, 1]
        
        return state_vars, p_disrupt
