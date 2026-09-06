import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from src.data.model502_schema import NUM_STATE_FIELDS, validate_state_tensor

class JorekGNNEncoder(nn.Module):
    """
    Perception (GNN Encoder) - Geometry-Aware Upgrade: 
    从 JOREK 的原生非结构化 Bezier 有限元网格中提取空间拓扑特征。
    使用 GATConv 并结合 edge_attr (Delta R, Delta Z) 来打破纯拓扑的对称性，
    从而更好地拟合台基区 (Pedestal) 的陡峭梯度和磁轴的剧烈变化。
    """
    def __init__(self, input_dim=NUM_STATE_FIELDS, hidden_dim=64, latent_dim=32, num_layers=3, edge_dim=2):
        super(JorekGNNEncoder, self).__init__()
        if input_dim != NUM_STATE_FIELDS:
            raise ValueError(
                f"Model 502 encoder requires input_dim={NUM_STATE_FIELDS}, got {input_dim}. "
                "Use an explicit legacy adapter for five-field checkpoints."
            )
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # GATConv 参数
        heads = 2
        concat = True
        
        # 输入层
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, concat=concat, edge_dim=edge_dim))
        self.norms.append(LayerNorm(hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=concat, edge_dim=edge_dim))
            self.norms.append(LayerNorm(hidden_dim))
            
        # 输出层 (Latent space)
        # 最后一层通常不拼接 heads，而是取平均 (concat=False)
        self.convs.append(GATConv(hidden_dim, latent_dim, heads=heads, concat=False, edge_dim=edge_dim))
        self.norms.append(LayerNorm(latent_dim))

    def forward(self, x, edge_index, pos):
        """
        Args:
            x: 节点特征矩阵 [Total_nodes, 9] in Model 502 canonical order
            edge_index: 图连接矩阵 [2, Num_edges]
            pos: 节点坐标矩阵 [Total_nodes, 2] (R, Z)
        Returns:
            latent_x: 潜在节点特征 [Total_nodes, latent_dim]
        """
        validate_state_tensor(x, name="encoder input")
        
        # 计算几何特征 (Delta R, Delta Z) 作为 edge_attr
        row, col = edge_index
        edge_attr = pos[row] - pos[col] # [Num_edges, 2]
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = F.silu(x) # SiLU (Swish) 激活函数在连续物理场建模中通常表现更好
            
        # 最后一层不加激活函数，保留完整的线性潜在空间
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        latent_x = self.norms[-1](x)
        
        return latent_x
