import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm

class JorekGNNEncoder(nn.Module):
    """
    Perception (GNN Encoder): 
    从 JOREK 的原生非结构化 Bezier 有限元网格中提取空间拓扑特征。
    """
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=32, num_layers=3):
        super(JorekGNNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 输入层
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
            
        # 输出层 (Latent space)
        self.convs.append(SAGEConv(hidden_dim, latent_dim))
        self.norms.append(LayerNorm(latent_dim))

    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征矩阵 [Total_nodes, 5] (psi, Phi, rho, T, v_parallel)
            edge_index: 图连接矩阵 [2, Num_edges]
        Returns:
            latent_x: 潜在节点特征 [Total_nodes, latent_dim]
        """
        # 防御性编程：严格验证输入特征维度是否符合 Reduced MHD Ansatz (5个变量)
        assert x.shape[1] == 5, f"Expected 5 input features (Reduced MHD), got {x.shape[1]}"
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.silu(x) # SiLU (Swish) 激活函数在连续物理场建模中通常表现更好
            
        # 最后一层不加激活函数，保留完整的线性潜在空间
        x = self.convs[-1](x, edge_index)
        latent_x = self.norms[-1](x)
        
        return latent_x
