import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from torch_geometric.nn import GATConv, LayerNorm
from src.data.model502_schema import NUM_STATE_FIELDS, validate_state_tensor


def _conv_block(conv, norm, x, edge_index, edge_attr):
    """One message-passing block: GAT + LayerNorm + SiLU."""
    return F.silu(norm(conv(x, edge_index, edge_attr=edge_attr)))


def _final_conv_block(conv, norm, x, edge_index, edge_attr):
    """Final message-passing block without the activation."""
    return norm(conv(x, edge_index, edge_attr=edge_attr))

class JorekGNNEncoder(nn.Module):
    """
    Perception (GNN Encoder) - Geometry-Aware Upgrade: 
    从 JOREK 的原生非结构化 Bezier 有限元网格中提取空间拓扑特征。
    使用 GATConv 并结合 edge_attr (Delta R, Delta Z) 来打破纯拓扑的对称性，
    从而更好地拟合台基区 (Pedestal) 的陡峭梯度和磁轴的剧烈变化。
    """
    def __init__(
        self,
        input_dim=NUM_STATE_FIELDS,
        hidden_dim=64,
        latent_dim=32,
        num_layers=3,
        edge_dim=2,
        checkpoint=False,
        condition_dim=0,
    ):
        super(JorekGNNEncoder, self).__init__()
        if input_dim != NUM_STATE_FIELDS:
            raise ValueError(
                f"Model 502 encoder requires input_dim={NUM_STATE_FIELDS}, got {input_dim}. "
                "Use an explicit legacy adapter for five-field checkpoints."
            )
        
        self.num_layers = num_layers
        self.checkpoint = bool(checkpoint)
        self.condition_dim = int(condition_dim)
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

        self.condition_mlp = None
        if self.condition_dim > 0:
            self.condition_mlp = nn.Sequential(
                nn.Linear(self.condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, x, edge_index, pos, condition=None, batch=None):
        """
        Args:
            x: 节点特征矩阵 [Total_nodes, 9] in Model 502 canonical order
            edge_index: 图连接矩阵 [2, Num_edges]
            pos: 节点坐标矩阵 [Total_nodes, 2] (R, Z)
        Returns:
            latent_x: 潜在节点特征 [Total_nodes, latent_dim]
            condition: 可选的零维条件向量 [Batch, condition_dim]，注入点为 encoder 输入
            batch: PyG batch 索引，用于把条件广播到各图的节点
        """
        validate_state_tensor(x, name="encoder input")
        if condition is not None:
            if self.condition_mlp is None:
                raise ValueError("encoder built without condition_dim cannot take a condition")
            condition = torch.as_tensor(condition, dtype=x.dtype, device=x.device)
            if condition.dim() == 1:
                condition = condition.view(1, -1)
            if condition.shape[1] != self.condition_dim:
                raise ValueError(
                    f"condition must have {self.condition_dim} entries, got {condition.shape[1]}"
                )
            embedding = self.condition_mlp(condition)
            if batch is not None:
                x = x + embedding[batch]
            else:
                x = x + embedding.expand(x.size(0), -1)
        
        # 计算几何特征 (Delta R, Delta Z) 作为 edge_attr
        row, col = edge_index
        edge_attr = pos[row] - pos[col] # [Num_edges, 2]
        
        use_checkpoint = self.checkpoint and self.training
        for i in range(self.num_layers - 1):
            if use_checkpoint:
                x = _torch_checkpoint(
                    _conv_block,
                    self.convs[i], self.norms[i], x, edge_index, edge_attr,
                    use_reentrant=False,
                )
            else:
                x = _conv_block(self.convs[i], self.norms[i], x, edge_index, edge_attr)
        
        # 最后一层不加激活函数，保留完整的线性潜在空间
        if use_checkpoint:
            latent_x = _torch_checkpoint(
                _final_conv_block,
                self.convs[-1], self.norms[-1], x, edge_index, edge_attr,
                use_reentrant=False,
            )
        else:
            latent_x = _final_conv_block(self.convs[-1], self.norms[-1], x, edge_index, edge_attr)
        
        return latent_x
