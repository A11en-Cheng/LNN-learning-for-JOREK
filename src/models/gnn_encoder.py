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
        # Explicit time conditioning: the rate function's contract is
        # rate(x, condition, source, dt), so log(dt_scaled) is injected as a
        # per-node feature (log covers the ~500x non-uniform dt range).
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Explicit history conditioning: the previous frame's per-channel
        # rate (already normalised by the decoder rate_scale) carries the
        # short-horizon trajectory information that a single snapshot lacks -
        # a supervised probe recovered Phi's rate only once history was added.
        self.history_mlp = nn.Sequential(
            nn.Linear(NUM_STATE_FIELDS, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Zero-init the residual branches: a checkpoint that predates these
        # inputs (loaded with strict=False) then behaves exactly as before
        # instead of injecting random noise.
        nn.init.zeros_(self.history_mlp[-1].weight)
        nn.init.zeros_(self.history_mlp[-1].bias)
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(
        self,
        x,
        edge_index,
        pos,
        condition=None,
        batch=None,
        time_input=None,
        history=None,
    ):
        """
        Args:
            x: 节点特征矩阵 [Total_nodes, 9] in Model 502 canonical order
            edge_index: 图连接矩阵 [2, Num_edges]
            pos: 节点坐标矩阵 [Total_nodes, 2] (R, Z)
        Returns:
            latent_x: 潜在节点特征 [Total_nodes, latent_dim]
            condition: 可选的零维条件向量 [Batch, condition_dim]，注入点为 encoder 输入
            batch: PyG batch 索引，用于把条件广播到各图的节点
            time_input: 可选逐节点 log(dt_scaled) [Total_nodes, 1]
            history: 可选逐节点上一帧速率 [Total_nodes, 9]
                （已按 rate_scale 归一化为 O(1)）
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
        if time_input is not None:
            time_input = torch.as_tensor(time_input, dtype=x.dtype, device=x.device)
            if time_input.dim() == 1:
                time_input = time_input.view(-1, 1)
            if time_input.size(0) != x.size(0):
                raise ValueError(
                    f"time_input must have one entry per node, got {time_input.size(0)}"
                )
            x = x + self.time_mlp(time_input)
        if history is not None:
            history = torch.as_tensor(history, dtype=x.dtype, device=x.device)
            if history.dim() == 1:
                history = history.view(-1, NUM_STATE_FIELDS)
            if history.size(0) != x.size(0) or history.size(1) != NUM_STATE_FIELDS:
                raise ValueError(
                    f"history must be [{x.size(0)}, {NUM_STATE_FIELDS}], "
                    f"got {tuple(history.shape)}"
                )
            x = x + self.history_mlp(history)
        
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
