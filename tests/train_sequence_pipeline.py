import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import JorekSequenceDataset, collate_temporal_graphs
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder
from src.physics.hard_constraints import ReducedMHDConstraints
from src.models.loss import JorekLoss
from src.utils.visualization import plot_jorek_field

def train_sequence_pipeline():
    # ================= 配置参数 =================
    DATA_DIR = "data/test_data_for_pipeline_EHL-2"
    OUTPUT_DIR = "output/sequence_training"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    VIS_DIR = os.path.join(OUTPUT_DIR, "vis")
    
    BATCH_SIZE = 2 # 由于图数据较大，Batch Size 不宜过大
    SEQ_LEN = 5    # 序列长度 T：输入 t0...t3，预测 t1...t4
    EPOCHS = 3
    LR = 1e-4
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"==================================================")
    print(f"🚀 Starting Sequence Training Pipeline on EHL-2 Data")
    print(f"   Batch: {BATCH_SIZE}, Seq Len: {SEQ_LEN}, Epochs: {EPOCHS}")
    print(f"==================================================")

    # ================= 1. 数据集准备 =================
    print("\n[1/5] Initializing Dataset...")
    dataset = JorekSequenceDataset(root_dir=DATA_DIR, seq_len=SEQ_LEN)
    
    # 验证数据量
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check data directory.")
        return

    # 切分训练集和验证集 (简单前 80% / 后 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   Total Samples: {len(dataset)}")
    print(f"   Train Samples: {len(train_dataset)}")
    print(f"   Val Samples:   {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_temporal_graphs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_temporal_graphs)

    # ================= 2. 模型构建 =================
    print("\n[2/5] Building GNN-LNN-GNN Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    encoder = JorekGNNEncoder(input_dim=5, hidden_dim=64, latent_dim=32, num_layers=3).to(device)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64, num_layers=1).to(device)
    decoder = JorekGNNDecoder(latent_dim=32, hidden_dim=64, output_dim=5, num_layers=3).to(device)
    
    # 我们将所有参数放入优化器
    # Reduced LR to 1e-5 for stability
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(processor.parameters()) + list(decoder.parameters()),
        lr=1e-5, weight_decay=1e-5
    )
    
    criterion = JorekLoss(lambda_reg=1.0, lambda_cls=0.1)

    # ================= 3. 训练循环 =================
    print("\n[3/5] Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        encoder.train(); processor.train(); decoder.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch_sequence in progress_bar:
            # batch_sequence 是一个长度为 SEQ_LEN 的列表，每个元素是一个 PyG Batch 对象
            # 输入序列: t=0 到 t=T-2
            # 目标序列: t=1 到 t=T-1
            optimizer.zero_grad()
            
            loss_accum = 0
            hidden_state = None # LNN 初始状态
            
            # Pre-process batch sequence (Normalization) on device
            # Avoid in-place modification inside the loop to prevent autograd errors
            # Pre-process batch sequence (Normalization) on device
            processed_sequence = []
            for data in batch_sequence:
                d = data.to(device)
                if hasattr(d, 'x'):
                    # Clone to avoid in-place modification of potentially cached data
                    new_x = d.x.clone()
                    # Psi (idx 0): [-10, 0] -> [-1, 0]
                    new_x[:, 0] = new_x[:, 0] * 0.1
                    d.x = new_x
                processed_sequence.append(d)

            # 时序循环 (BPTT)
            # 我们迭代 0 到 SEQ_LEN-2 (即预测下一步)
            for t in range(SEQ_LEN - 1):
                current_data = processed_sequence[t]
                next_data_target = processed_sequence[t+1]
                
                # 1. Perception
                latent_x = encoder(current_data.x, current_data.edge_index)
                
                # 2. Cognition (Recurrent Update)
                # 传入 hidden_state 并在下一步复用
                next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
                
                # 3. Reconstruction
                pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch)
                
                # 4. Loss Calculation
                # 假设 target_p_disrupt 暂时全为 0 (无破裂标签数据时)
                # 实际项目中应从 current_data 中读取破裂标签
                # Warning: num_graphs 可能是一个 batch 的 batch_size
                dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)
                
                # Check for NaNs in predictions
                if torch.isnan(pred_x).any() or torch.isnan(pred_p).any():
                    print(f"NaN detected in predictions at step {t}")
                    print(f"pred_x stats: min={pred_x.min()}, max={pred_x.max()}")
                    print(f"pred_p stats: min={pred_p.min()}, max={pred_p.max()}")
                    # Abort this batch to avoid crashing
                    loss_accum = torch.tensor(0.0, device=device, requires_grad=True)
                    break

                step_loss, _, _ = criterion(pred_x, next_data_target.x, pred_p, dummy_target_p)
                loss_accum += step_loss
            
            if loss_accum.item() == 0.0:
                continue

            # 平均时间步损失
            avg_seq_loss = loss_accum / (SEQ_LEN - 1)
            avg_seq_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(processor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            
            optimizer.step()
            
            total_train_loss += avg_seq_loss.item()
            progress_bar.set_postfix({'loss': avg_seq_loss.item()})
            
        print(f"   Epoch {epoch+1} Train Loss: {total_train_loss / len(train_loader):.6f}")
        
        # --- Validation Phase ---
        # (简单起见，这里只跑验证循环，不更新梯度)
        encoder.eval(); processor.eval(); decoder.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_sequence in val_loader:
                # Pre-process batch sequence (Normalization) on device
                processed_sequence = []
                for data in batch_sequence:
                    d = data.to(device)
                    if hasattr(d, 'x'):
                        # Clone to avoid in-place modification
                        new_x = d.x.clone()
                        new_x[:, 0] = new_x[:, 0] * 0.1
                        d.x = new_x
                    processed_sequence.append(d)

                hidden_state = None
                loss_accum = 0
                for t in range(SEQ_LEN - 1):
                    current_data = processed_sequence[t]
                    next_data_target = processed_sequence[t+1]
                    
                    latent_x = encoder(current_data.x, current_data.edge_index)
                    next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
                    pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch)
                    
                    dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)

                    if torch.isnan(pred_x).any() or torch.isnan(pred_p).any():
                        break
                    
                    step_loss, _, _ = criterion(pred_x, next_data_target.x, pred_p, dummy_target_p)
                    loss_accum += step_loss
                total_val_loss += (loss_accum / (SEQ_LEN - 1)).item()
                
        print(f"   Epoch {epoch+1} Val Loss:   {total_val_loss / len(val_loader):.6f}")

        # ================= 4. 保存检查点 =================
        checkpoint_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'processor_state_dict': processor.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"   Saved Checkpoint: {checkpoint_path}")

    # ================= 5. 可视化测试结果 =================
    print("\n[5/5] Generating Prediction Visualization...")
    # 取验证集的第一个序列进行可视化
    sample_seq = next(iter(val_loader))
    
    # 预测序列的最后一步
    t_inspect = SEQ_LEN - 2 # 最后一个预测步 (对应 SEQ_LEN-1 的目标)
    input_data = sample_seq[t_inspect].to(device)
    target_data = sample_seq[t_inspect+1].to(device)
    
    encoder.eval(); processor.eval(); decoder.eval()
    with torch.no_grad():
        # 这里仅做单步推理演示，实际应跑完整个序列状态
        latent_x = encoder(input_data.x, input_data.edge_index)
        next_latent_x, _ = processor(latent_x, input_data.batch) # 简化：无状态
        pred_x, _ = decoder(next_latent_x, input_data.edge_index, input_data.batch)

    # 取 Batch 中的第一个图
    # 节点特征 [Total_Nodes, 5] -> 也就是 [Batch * Nodes, 5]
    # 我们知道 batch=2，所以需要切分
    # ptr 是每张图在 x 中的起始索引 [0, N1, N1+N2, ...]
    if hasattr(input_data, 'ptr'):
        num_nodes = input_data.ptr[1] - input_data.ptr[0]
    else:
        # 如果没有 ptr，假定均匀分布
        num_nodes = input_data.x.shape[0] // input_data.batch_size
        
    pred_x_sample = pred_x[:num_nodes, :]
    truth_x_sample = target_data.x[:num_nodes, :]
    pos_sample = input_data.pos[:num_nodes, :]
    # 这里我们不处理 batching 的 triangles，简单地不做凹变形处理，只看初步结果
    vis_triangles = None 

    # 绘制 Flux (psi) - Channel 0
    try:
        plot_jorek_field(
            pos=pos_sample.cpu(), 
            values=pred_x_sample[:, 0].cpu(),
            triangles=vis_triangles, 
            title=f"Predicted Psi (Step {t_inspect})", 
            save_path=os.path.join(VIS_DIR, "pred_psi.png")
        )
        
        plot_jorek_field(
            pos=pos_sample.cpu(), 
            values=truth_x_sample[:, 0].cpu(), 
            triangles=vis_triangles,
            title=f"Ground Truth Psi (Step {t_inspect})", 
            save_path=os.path.join(VIS_DIR, "truth_psi.png")
        )
        print(f"✅ Training Pipeline Completed! Results in {OUTPUT_DIR}")
    except ValueError as e:
        print(f"⚠️ Visualization failed due to NaN/Inf in predictions: {e}")
        print("This indicates model divergence. Please adjust hyperparameters (LR, Norm) or check data.")
        print(f"Model checkpoints saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    train_sequence_pipeline()
