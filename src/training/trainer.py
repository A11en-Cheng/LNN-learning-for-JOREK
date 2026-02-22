import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

class JorekTrainer:
    """
    训练循环管理器。
    负责前向传播、损失计算、反向传播和梯度裁剪。
    """
    def __init__(self, encoder, processor, decoder, physics_layer, criterion, lr=1e-3, clip_value=1.0):
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.physics_layer = physics_layer
        self.criterion = criterion
        self.clip_value = clip_value
        
        # 将所有模型参数传递给优化器
        self.optimizer = AdamW(
            list(self.encoder.parameters()) + 
            list(self.processor.parameters()) + 
            list(self.decoder.parameters()),
            lr=lr,
            weight_decay=1e-4 # L2 正则化，防止过拟合
        )

    def train_step(self, batch_data, target_states, target_p_disrupt):
        """
        执行单步训练。
        Args:
            batch_data: 当前时间步的 PyG Batch 数据 (包含 x, edge_index, pos, batch 等)
            target_states: 下一时间步的真实物理状态 [Total_nodes, 5]
            target_p_disrupt: 真实的破裂标签 [Batch_size, 1]
        Returns:
            total_loss, reg_loss, cls_loss
        """
        # 1. 设置为训练模式
        self.encoder.train()
        self.processor.train()
        self.decoder.train()
        
        # 2. 清空梯度
        self.optimizer.zero_grad()
        
        # 3. 前向传播 (Perception -> Cognition -> Reconstruction)
        latent_features = self.encoder(batch_data.x, batch_data.edge_index, batch_data.pos)
        # Processor 现在返回 (output, hidden_state)
        next_latent_features, _ = self.processor(latent_features, batch_data.batch)
        pred_states, pred_p_disrupt = self.decoder(next_latent_features, batch_data.edge_index, batch_data.batch, batch_data.pos)
        
        # 4. 计算损失
        from src.models.loss import compute_spatial_weights
        spatial_weights = compute_spatial_weights(batch_data.x[:, 0], batch_data.batch, axis_weight=10.0, pedestal_weight=5.0)
        total_loss, diagnostics = self.criterion(
            pred_states, target_states, pred_p_disrupt, target_p_disrupt, node_weights=spatial_weights
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
        
        return total_loss.item(), loss_reg.item(), loss_cls.item()

    def eval_step(self, batch_data, target_states, target_p_disrupt):
        """
        执行单步评估。
        """
        self.encoder.eval()
        self.processor.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            latent_features = self.encoder(batch_data.x, batch_data.edge_index, batch_data.pos)
            # Processor 现在返回 (output, hidden_state)
            next_latent_features, _ = self.processor(latent_features, batch_data.batch)
            pred_states, pred_p_disrupt = self.decoder(next_latent_features, batch_data.edge_index, batch_data.batch, batch_data.pos)
            
            from src.models.loss import compute_spatial_weights
            spatial_weights = compute_spatial_weights(batch_data.x[:, 0], batch_data.batch, axis_weight=10.0, pedestal_weight=5.0)
            total_loss, diagnostics = self.criterion(
                pred_states, target_states, pred_p_disrupt, target_p_disrupt, node_weights=spatial_weights
            )
            loss_reg = diagnostics.get("loss_reg", 0.0)
            loss_cls = diagnostics.get("loss_cls", 0.0)
            
        return total_loss.item(), loss_reg, loss_cls, pred_states, pred_p_disrupt
