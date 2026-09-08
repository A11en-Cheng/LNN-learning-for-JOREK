import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from src.data.model502_schema import validate_projection_tensor, validate_state_tensor
from src.data.dataset import step_dt_jorek
from src.models.loss import compute_spatial_weights


def _detach_state_struct(state):
    """Detach a (possibly nested) LNN state at a BPTT boundary."""
    if state is None:
        return None
    if isinstance(state, (tuple, list)):
        return tuple(
            item.detach() if isinstance(item, torch.Tensor) else item for item in state
        )
    if isinstance(state, torch.Tensor):
        return state.detach()
    return state


def _encoder_inputs(encoder, batch):
    """Feed the encoder, passing the condition only when it has a conditioning head."""
    condition = getattr(batch, "condition", None) if getattr(encoder, "condition_dim", 0) > 0 else None
    return batch.x, batch.edge_index, batch.pos, condition, batch.batch

class JorekTrainer:
    """
    训练循环管理器。
    负责前向传播、损失计算、反向传播和梯度裁剪。
    """
    def __init__(
        self,
        encoder,
        processor,
        decoder,
        physics_layer,
        criterion,
        lr=1e-3,
        clip_value=1.0,
        dt_ref=1.0,
    ):
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.physics_layer = physics_layer
        self.criterion = criterion
        self.clip_value = clip_value
        self.dt_ref = float(dt_ref)
        self.hidden_state = None
        
        # 将所有模型参数传递给优化器
        self.optimizer = AdamW(
            list(self.encoder.parameters()) + 
            list(self.processor.parameters()) + 
            list(self.decoder.parameters()),
            lr=lr,
            weight_decay=1e-4 # L2 正则化，防止过拟合
        )

    def reset_state(self):
        """Drop the carried LNN state; the next step starts a fresh trajectory."""
        self.hidden_state = None

    def detach_state(self):
        """Cut the graph at a BPTT segment boundary."""
        self.hidden_state = _detach_state_struct(self.hidden_state)

    def train_step(
        self,
        batch_data,
        target_states,
        target_p_disrupt,
        dt=1.0,
        controls=None,
        state=None,
        time_steps=None,
        detach_state_on_entry=True,
    ):
        """
        执行单步训练。
        Args:
            batch_data: 当前时间步的 PyG Batch 数据 (包含 x, edge_index, pos, batch 等)
            target_states: 下一时间步的真实物理状态 [Total_nodes, 9]
            target_p_disrupt: 真实的破裂标签 [Batch_size, 1]
            dt: 物理时间步（秒），同时传给 PDE 项（若 criterion 支持）
            state: 上一个时间步的 LNN 状态；None 表示新轨迹起点
            time_steps: 传给 CfC 的 timespan；缺省用 dt
            detach_state_on_entry: 入口处切断携带状态的计算图（BPTT 段边界）。
        Returns:
            total_loss, reg_loss, cls_loss
        """
        validate_state_tensor(batch_data.x, name="trainer current state")
        validate_state_tensor(target_states, name="trainer target state")
        controls = controls if controls is not None else getattr(batch_data, "source", None)
        if controls is not None:
            validate_projection_tensor(controls, name="trainer controls")

        # 1. 设置为训练模式
        self.encoder.train()
        self.processor.train()
        self.decoder.train()
        
        # 2. 清空梯度
        self.optimizer.zero_grad()
        
        # 3. 前向传播 (Perception -> Cognition -> Reconstruction)
        latent_features = self.encoder(*_encoder_inputs(self.encoder, batch_data))
        # Processor 现在返回 (output, hidden_state)
        state_in = state if state is not None else self.hidden_state
        if detach_state_on_entry:
            state_in = _detach_state_struct(state_in)
        next_latent_features, next_state = self.processor(
            latent_features,
            batch_data.batch,
            state=state_in,
            time_steps=dt if time_steps is None else time_steps,
            controls=controls,
        )
        self.hidden_state = next_state
        pred_delta, pred_p_disrupt = self.decoder(next_latent_features, batch_data.edge_index, batch_data.batch, batch_data.pos)
        pred_states = batch_data.x + pred_delta * (dt / self.dt_ref)
        
        # 4. 计算损失
        from src.models.loss import compute_spatial_weights
        spatial_weights = compute_spatial_weights(batch_data.x[:, 0], batch_data.batch, axis_weight=10.0, pedestal_weight=5.0)
        criterion_kwargs = {"dt": dt} if hasattr(self.criterion, "dt") else {}
        freeboundary = getattr(batch_data, "freeboundary", None)
        if freeboundary is None:
            freeboundary = True
        total_loss, diagnostics = self.criterion(
            pred_states,
            target_states,
            pred_p_disrupt,
            target_p_disrupt,
            node_weights=spatial_weights,
            batch_boundary=getattr(batch_data, "boundary", None),
            freeboundary=freeboundary,
            **criterion_kwargs,
        )
        loss_reg = diagnostics.get("loss_reg", 0.0)
        loss_cls = diagnostics.get("loss_cls", 0.0)
        
        # 5. 反向传播
        total_loss.backward()
        
        # 6. 梯度裁剪 (Gradient Clipping)
        # 关键步骤：防止在预测破裂等刚性动态时发生梯度爆炸
        clip_grad_norm_(self.encoder.parameters(), self.clip_value)
        clip_grad_norm_(self.processor.parameters(), self.clip_value)
        clip_grad_norm_(self.decoder.parameters(), self.clip_value)
        
        # 7. 更新参数
        self.optimizer.step()
        
        return total_loss.item(), float(loss_reg), float(loss_cls)

    def eval_step(
        self,
        batch_data,
        target_states,
        target_p_disrupt,
        dt=1.0,
        controls=None,
        state=None,
        time_steps=None,
        detach_state_on_entry=True,
    ):
        """
        执行单步评估。
        """
        self.encoder.eval()
        self.processor.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            latent_features = self.encoder(*_encoder_inputs(self.encoder, batch_data))
            # Processor 现在返回 (output, hidden_state)
            controls = controls if controls is not None else getattr(batch_data, "source", None)
            if controls is not None:
                validate_projection_tensor(controls, name="trainer controls")
            state_in = state if state is not None else self.hidden_state
            if detach_state_on_entry:
                state_in = _detach_state_struct(state_in)
            next_latent_features, next_state = self.processor(
                latent_features,
                batch_data.batch,
                state=state_in,
                time_steps=dt if time_steps is None else time_steps,
                controls=controls,
            )
            self.hidden_state = next_state
            pred_delta, pred_p_disrupt = self.decoder(next_latent_features, batch_data.edge_index, batch_data.batch, batch_data.pos)
            pred_states = batch_data.x + pred_delta * (dt / self.dt_ref)
            
            from src.models.loss import compute_spatial_weights
            spatial_weights = compute_spatial_weights(batch_data.x[:, 0], batch_data.batch, axis_weight=10.0, pedestal_weight=5.0)
            criterion_kwargs = {"dt": dt} if hasattr(self.criterion, "dt") else {}
            freeboundary = getattr(batch_data, "freeboundary", None)
            if freeboundary is None:
                freeboundary = True
            total_loss, diagnostics = self.criterion(
                pred_states,
                target_states,
                pred_p_disrupt,
                target_p_disrupt,
                node_weights=spatial_weights,
                batch_boundary=getattr(batch_data, "boundary", None),
                freeboundary=freeboundary,
                **criterion_kwargs,
            )
            loss_reg = diagnostics.get("loss_reg", 0.0)
            loss_cls = diagnostics.get("loss_cls", 0.0)
            
        return total_loss.item(), loss_reg, loss_cls, pred_states, pred_p_disrupt

    def train_step_multihorizon(
        self,
        batch_list,
        target_p_disrupt,
        horizons=(1, 2, 4, 8),
        controls_list=None,
        return_channels=False,
    ):
        """Train one window with skip-sampled horizons at their true summed dt.

        ``batch_list`` is a time-major list of PyG Batches.  For every pair
        ``(t, t+k)`` the model takes one step with timespan
        ``dt = sum of the per-frame dts between t and t+k`` and is compared
        against the frame at ``t+k``.  The one-step branch (k=1) carries the LNN
        state to the next frame; all longer horizons branch from the same state.
        The carried state is detached at the window boundary (BPTT segment).
        """
        self.encoder.train()
        self.processor.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        self.reset_state()

        total_loss = torch.zeros((), dtype=torch.float32)
        pair_count = 0
        channel_sums = {}
        for t, batch in enumerate(batch_list):
            state_t = self.hidden_state
            controls = (
                controls_list[t]
                if controls_list is not None
                else getattr(batch, "source", None)
            )
            for k in horizons:
                if t + k >= len(batch_list):
                    continue
                target = batch_list[t + k]
                dt = float(step_dt_jorek(batch, target).mean().item())
                latent = self.encoder(*_encoder_inputs(self.encoder, batch))
                next_latent, next_state = self.processor(
                    latent,
                    batch.batch,
                    state=state_t,
                    time_steps=dt,
                    controls=controls,
                )
                if k == 1:
                    self.hidden_state = next_state
                pred_delta, pred_p = self.decoder(
                    next_latent, batch.edge_index, batch.batch, batch.pos
                )
                pred_states = batch.x + pred_delta * (dt / self.dt_ref)
                spatial_weights = compute_spatial_weights(
                    batch.x[:, 0], batch.batch, axis_weight=10.0, pedestal_weight=5.0
                )
                criterion_kwargs = {"dt": dt} if hasattr(self.criterion, "dt") else {}
                freeboundary = getattr(batch, "freeboundary", None)
                if freeboundary is None:
                    freeboundary = True
                loss, diagnostics = self.criterion(
                    pred_states,
                    target.x,
                    pred_p,
                    target_p_disrupt,
                    node_weights=spatial_weights,
                    batch_boundary=getattr(batch, "boundary", None),
                    freeboundary=freeboundary,
                    **criterion_kwargs,
                )
                for key, value in diagnostics.items():
                    if (
                        isinstance(key, str)
                        and key.startswith("loss_")
                        and key not in {"loss_reg", "loss_cls", "loss_total"}
                    ):
                        channel_sums[key] = channel_sums.get(key, 0.0) + value
                total_loss = total_loss + loss
                pair_count += 1

        if pair_count == 0:
            raise ValueError("no (t, t+k) pair fits the window and horizons")
        total_loss = total_loss / pair_count
        total_loss.backward()
        clip_grad_norm_(self.encoder.parameters(), self.clip_value)
        clip_grad_norm_(self.processor.parameters(), self.clip_value)
        clip_grad_norm_(self.decoder.parameters(), self.clip_value)
        self.optimizer.step()
        self.detach_state()
        if return_channels:
            channel_avg = {key: value / pair_count for key, value in channel_sums.items()}
            return float(total_loss.item()), pair_count, channel_avg
        return float(total_loss.item()), pair_count
