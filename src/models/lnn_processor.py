import torch
import torch.nn as nn
from torch_geometric.utils import unbatch
from src.data.model502_schema import NUM_MARKER_PROJECTION_FIELDS, validate_projection_tensor
from src.models.ltc_cell import LTCCell


class JorekLTCProcessor(nn.Module):
    """Per-node LTC propagator built on :class:`~src.models.ltc_cell.LTCCell`.

    The cell predicts the local decay ``rate`` (softplus -> strictly positive)
    and the equilibrium ``target`` (tanh) and applies the exact solution of
    ``dh/dt = -rate * (h - target)`` over the requested timespan - see the
    module docstring of ``src/models/ltc_cell.py`` for the identity/semigroup
    guarantees.

    Replaces two things that made the previous cell unfit as a time propagator:

    * the ncps update ``h_new = ff1*(1-t) + t*ff2`` (not the identity at
      ``ts = 0``, ~21% of an update of split ambiguity);
    * the ``mixed_memory`` LSTM shortcut, which ignores ``ts`` entirely.

    Interface-identical to :class:`JorekLNNProcessor`, so ``CfCInterpolator``
    can swap processors with a flag.
    """

    def __init__(
        self,
        latent_dim=32,
        hidden_size=64,
        backbone_units: int = 256,
        backbone_layers: int = 2,
        state_dependent: bool = False,
        target_activation: str = "tanh",
        init_decay: float = 0.5,
        ts_reference: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.control_encoder = nn.Sequential(
            nn.Linear(NUM_MARKER_PROJECTION_FIELDS, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.cell = LTCCell(
            latent_dim, hidden_size,
            backbone_units=backbone_units, backbone_layers=backbone_layers,
            state_dependent=state_dependent, target_activation=target_activation,
            init_decay=init_decay,
            ts_reference=ts_reference,
        )
        self.proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, latent_x, batch_index=None, state=None, time_steps=None, controls=None):
        if controls is not None:
            validate_projection_tensor(controls, name="processor controls")
            if controls.shape[0] != latent_x.shape[0]:
                raise ValueError(
                    "processor controls and latent_x must have the same number of nodes: "
                    f"{controls.shape[0]} != {latent_x.shape[0]}"
                )
            latent_x = latent_x + self.control_encoder(controls.to(latent_x.dtype))

        total_nodes = latent_x.size(0)
        batch_size = 1 if batch_index is None else int(batch_index.max().item()) + 1
        if total_nodes % batch_size != 0:
            raise ValueError(
                "LTC processor requires a fixed mesh size across the batch "
                f"(got {total_nodes} rows for {batch_size} graphs)"
            )
        if state is None:
            h_state = torch.zeros(
                (total_nodes, self.hidden_size), device=latent_x.device, dtype=latent_x.dtype
            )
        elif isinstance(state, tuple):  # tolerate a (h, c) state from the ncps path
            h_state = state[0]
        else:
            h_state = state

        if time_steps is None:
            steps = torch.ones((total_nodes, 1), device=latent_x.device, dtype=latent_x.dtype)
        else:
            steps = torch.as_tensor(time_steps, dtype=latent_x.dtype, device=latent_x.device)
            if steps.numel() == 1:
                steps = steps.reshape(1, 1).expand(total_nodes, 1)
            elif batch_index is not None and steps.numel() == batch_size:
                steps = steps.reshape(-1)[batch_index].view(total_nodes, 1)
            else:
                raise ValueError(
                    "time_steps must be a scalar or one entry per graph; "
                    f"got shape {tuple(steps.shape)} for batch_size {batch_size}"
                )

        next_state = self.cell(latent_x, h_state, steps)
        return self.proj(next_state), next_state


class JorekLNNProcessor(nn.Module):
    """
    Cognition (LNN Processor):
    使用 Liquid Neural Networks (CfC) 学习潜在空间中连续时间的动态演化。
    专门用于处理 JOREK 模拟中的刚性动态（如热猝灭和破裂）。

    .. warning::
       LEGACY path (``--cell ncps``).  The installed ncps cell is *not* a time
       propagator (``ts = 0`` is not the identity, pitfall #43) and
       ``mixed_memory=True`` adds an LSTM shortcut that ignores ``ts``
       completely, so mixed memory is off by default.  Only pass
       ``mixed_memory=True`` to re-load checkpoints trained before 2026-10-01
       (e.g. ``gate6_deriv2``, ``long60x20``).
    """
    def __init__(self, latent_dim=32, hidden_size=64, num_layers=1, mixed_memory=False):
        super(JorekLNNProcessor, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        # Node-local, pre-standardised Model 502 source controls are injected
        # immediately before continuous-time evolution.  ``None`` means a
        # genuinely unforced trajectory; callers must not replace unavailable
        # source data with zeros.
        self.control_encoder = nn.Sequential(
            nn.Linear(NUM_MARKER_PROJECTION_FIELDS, latent_dim),
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
                mixed_memory=bool(mixed_memory),
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
            controls: 可选的、已归一化 Model 502 五场投影 ``[Total_nodes, 5]``，
                      顺序为 ``[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]``。
        Returns:
            next_latent_output: LNN 的输出特征 [Total_nodes, latent_dim]
            next_state: 更新后的隐藏状态
        """
        if controls is not None:
            validate_projection_tensor(controls, name="processor controls")
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
            # 这里将 time_steps 解释为“当前时间步的物理时间间隔 Δt”。
            timespans = None
            if time_steps is not None:
                if not isinstance(time_steps, torch.Tensor):
                    time_steps = torch.tensor(time_steps, dtype=seq_x.dtype, device=seq_x.device)
                else:
                    time_steps = time_steps.to(dtype=seq_x.dtype, device=seq_x.device)
                # A scalar is shared by all samples.  A vector with one entry
                # per graph is expanded to every node of that graph, matching
                # the graph-major ordering of the [B*N, 1, F] sequence tensor,
                # so batched trajectories keep their true non-uniform dt.
                if time_steps.numel() == 1:
                    timespans = time_steps.view(1, 1)
                elif batch_index is not None and time_steps.numel() == batch_size:
                    timespans = time_steps.reshape(-1)[batch_index].view(total_nodes, 1)
                else:
                    raise ValueError(
                        "time_steps must be a scalar or one entry per graph; "
                        f"got shape {tuple(time_steps.shape)} for batch_size {batch_size}"
                    )
            # The installed ncps CfC wrapper does ``timespans[:, t].squeeze()``,
            # which collapses a per-sample timespan to shape [B] and cannot
            # broadcast against the cell's [B, units] time gate.  Run the
            # single-step cell directly with a [B, 1] timespan instead, so each
            # node keeps its graph's true non-uniform dt.
            inputs = seq_x[:, 0, :]
            if state is None:
                h_state = torch.zeros(
                    (total_nodes, self.rnn.state_size),
                    device=seq_x.device,
                    dtype=seq_x.dtype,
                )
                c_state = torch.zeros_like(h_state) if self.rnn.use_mixed else None
            elif self.rnn.use_mixed:
                h_state, c_state = state
            else:
                h_state, c_state = state, None
            ts = (
                torch.ones((total_nodes, 1), device=seq_x.device, dtype=seq_x.dtype)
                if timespans is None
                else timespans[:, 0].view(-1, 1)
            )
            if self.rnn.use_mixed:
                h_state, c_state = self.rnn.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn.rnn_cell.forward(inputs, h_state, ts)
            output = self.rnn.fc(h_out).unsqueeze(1)
            next_state = (h_state, c_state) if self.rnn.use_mixed else h_state
        else:
            # GRU returns: output, h_n
            output, next_state = self.rnn(seq_x, state)
            
        # 4. Sequence-to-Graph 转换 (还原形状)
        # out 形状: [Batch_size * Num_nodes, Seq_len=1, latent_dim] -> [Total_nodes, latent_dim]
        next_latent_output = output.view(total_nodes, feat_dim)
        
        return next_latent_output, next_state
