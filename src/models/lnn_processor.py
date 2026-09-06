import torch
import torch.nn as nn
from torch_geometric.utils import unbatch
from src.data.model502_schema import NUM_SOURCE_FIELDS, validate_source_tensor

class JorekLNNProcessor(nn.Module):
    """
    Cognition (LNN Processor):
    使用 Liquid Neural Networks (CfC) 学习潜在空间中连续时间的动态演化。
    专门用于处理 JOREK 模拟中的刚性动态（如热猝灭和破裂）。
    """
    def __init__(self, latent_dim=32, hidden_size=64, num_layers=1):
        super(JorekLNNProcessor, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        # Node-local, pre-standardised Model 502 source controls are injected
        # immediately before continuous-time evolution.  ``None`` means a
        # genuinely unforced trajectory; callers must not replace unavailable
        # source data with zeros.
        self.control_encoder = nn.Sequential(
            nn.Linear(NUM_SOURCE_FIELDS, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # 尝试导入 ncps (Liquid Neural Networks 库)
        try:
            from ncps.torch import CfC
            # CfC (Closed-form Continuous-time) 是一种高效的 LNN 实现
            self.rnn = CfC(
                input_size=latent_dim,
                units=hidden_size,
                proj_size=latent_dim, # 投影回潜在维度
                batch_first=True,     # 期望输入形状: [Batch, Seq_len, Features]
                mixed_memory=True     # 混合记忆模式通常表现更好
            )
            self.use_cfc = True
        except ImportError:
            print("Warning: 'ncps' package not found. Falling back to standard GRU for testing.")
            print("Please install it via: pip install ncps")
            # 降级方案：如果未安装 ncps，使用 GRU 代替以便测试管道
            self.rnn = nn.GRU(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                batch_first=True
            )
            self.use_cfc = False

    def forward(self, latent_x, batch_index=None, state=None, time_steps=None, controls=None):
        """
        Args:
            latent_x: GNN 编码器输出的潜在特征 [Total_nodes, latent_dim]
            batch_index: PyG 的 batch 索引向量 [Total_nodes]
            state: RNN/LNN 的隐藏状态 (None for init)
            time_steps: 时间步长信息 (可选)。
                        推荐传入一个标量 Δt（物理时间间隔），
                        LNN 将在当前时间步对所有节点使用相同的 Δt 进行连续时间积分。
            controls: 可选的、已归一化 Model 502 源项 ``[Total_nodes, 3]``，
                      顺序为 ``[S_rho, S_T_i, S_T_e]``。
        Returns:
            next_latent_output: LNN 的输出特征 [Total_nodes, latent_dim]
            next_state: 更新后的隐藏状态
        """
        if controls is not None:
            validate_source_tensor(controls, name="processor controls")
            if controls.shape[0] != latent_x.shape[0]:
                raise ValueError(
                    "processor controls and latent_x must have the same number of nodes: "
                    f"{controls.shape[0]} != {latent_x.shape[0]}"
                )
            latent_x = latent_x + self.control_encoder(controls.to(latent_x.dtype))

        # 1. CRITICAL TENSOR SHAPE RULE: Graph-to-Sequence 转换
        # PyG 将多个图拼接为 [Total_nodes, Feat]。
        # LNN 期望的输入是 [Batch_size, Seq_len, Feat]。
        # 在这个架构中，我们将每个节点视为一个独立的序列。
        
        # 还原为批次维度
        # 注意：这里假设每个图的节点数相同（对于同一个 JOREK 网格通常是成立的）
        # 将 PyG Batch 转换为 [Batch_Size, Num_Nodes, Feat]
        
        if batch_index is None:
            batch_size = 1
        else:
            batch_size = batch_index.max().item() + 1
            
        total_nodes = latent_x.size(0)
        num_nodes = total_nodes // batch_size
        feat_dim = latent_x.size(1)

        # 检查是否整除，如果不整除说明图大小不一致，这种情况下无法简单 reshape 为 tensor
        # 需要 pad packed sequence，但这里为了性能优化，我们假设同伦网格
        if total_nodes % batch_size != 0:
             # Fallback to unbatch (slower but safer)
             # 但 RNN 需要 tensor 输入，不能是 list
             # 如果节点数不同，只能 pad
             # 暂时这部分仅支持固定拓扑
             raise ValueError("LNN Processor currently requires fixed mesh size across batch for efficient tensor operations.")
             
        # [Batch, Nodes, Feat] -> [Batch * Nodes, 1, Feat]
        # 将 (Batch, Node) 视为独立的序列样本
        seq_x = latent_x.view(batch_size * num_nodes, 1, feat_dim)
        
        # 3. LNN 前向传播
        if self.use_cfc:
            # CfC returns: output, state
            # 这里将 time_steps 解释为“当前时间步的物理时间间隔 Δt”
            # 为了与 ncps.CfC 的接口兼容，使用 shape=[1, 1] 的 timespans，
            # 使其在内部被视为对所有 batch 元素共享的标量 Δt。
            timespans = None
            if time_steps is not None:
                if not isinstance(time_steps, torch.Tensor):
                    time_steps = torch.tensor(time_steps, dtype=seq_x.dtype, device=seq_x.device)
                # 标量 Δt 或形如 [1] 的向量都折叠为 [1,1]
                if time_steps.dim() == 0:
                    timespans = time_steps.view(1, 1)
                elif time_steps.dim() == 1 and time_steps.numel() == 1:
                    timespans = time_steps.view(1, 1)
                else:
                    # 如果传入了更高维度的 dt，这里退化为取其均值作为全局 Δt
                    timespans = time_steps.mean().view(1, 1)
            output, next_state = self.rnn(seq_x, state, timespans=timespans)
        else:
            # GRU returns: output, h_n
            output, next_state = self.rnn(seq_x, state)
            
        # 4. Sequence-to-Graph 转换 (还原形状)
        # out 形状: [Batch_size * Num_nodes, Seq_len=1, latent_dim] -> [Total_nodes, latent_dim]
        next_latent_output = output.view(total_nodes, feat_dim)
        
        return next_latent_output, next_state
