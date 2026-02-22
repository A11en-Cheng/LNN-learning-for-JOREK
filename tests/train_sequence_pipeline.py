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
from src.models.loss import JorekLoss, compute_spatial_weights
from src.utils.visualization import plot_jorek_field

def train_sequence_pipeline():
    # ================= 配置参数 (GPU Optimized) =================
    DATA_DIR = "data/test_data_for_pipeline_EHL-2"
    OUTPUT_DIR = "output/sequence_training"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    VIS_DIR = os.path.join(OUTPUT_DIR, "vis")
    
    # 针对 RTX 5070 Ti 激进调优
    BATCH_SIZE = 8    # 增加 Batch Size 提高并行度 (From 4 -> 16)
    SEQ_LEN = 10       # 保持适中的序列长度，或根据需要增加，但 Batch 优先
    EPOCHS = 200       # 训练更多轮次
    LR = 2e-4         # 稍微增大 LR 配合更大的 Batch Size
    
    # 模型规模参数
    HIDDEN_DIM = 256   # 增加网络宽度 (From 128 -> 256)
    LATENT_DIM = 64    # 增加潜在空间 (From 32 -> 64)
    
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"==================================================")
    print(f"🚀 Starting Sequence Training Pipeline on EHL-2 Data (GPU Optimized)")
    print(f"   Batch: {BATCH_SIZE}, Seq Len: {SEQ_LEN}, Epochs: {EPOCHS}")
    print(f"   Hidden: {HIDDEN_DIM}, Latent: {LATENT_DIM}")
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
    
    # Num Workers 增加数据加载并行度
    num_workers = min(8, os.cpu_count()) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_temporal_graphs, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_temporal_graphs, num_workers=num_workers, persistent_workers=True)

    # ================= 2. 模型构建 =================
    print("\n[2/5] Building GNN-LNN-GNN Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # 使用更大的模型规模
    encoder = JorekGNNEncoder(input_dim=5, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=4).to(device) # Layers 3->4
    processor = JorekLNNProcessor(latent_dim=LATENT_DIM, hidden_size=HIDDEN_DIM, num_layers=2).to(device) # Layers 1->2
    decoder = JorekGNNDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=5, num_layers=4).to(device) # Layers 3->4
    
    # 我们将所有参数放入优化器
    # Reduced LR to 1e-5 for stability
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(processor.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=1e-5
    )
    
    criterion = JorekLoss(lambda_reg=1.0, lambda_cls=0.1)

    
    # Loss History
    loss_history = {
        "train_loss": [],
        "val_loss": []
    }
    
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
                latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)
                
                # 2. Cognition (Recurrent Update)
                # 传入 hidden_state 并在下一步复用
                next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
                
                # 3. Reconstruction
                pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch, current_data.pos)
                
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

                # 计算空间权重 (Spatial Weighting)
                spatial_weights = compute_spatial_weights(current_data.x[:, 0], current_data.batch, axis_weight=10.0, pedestal_weight=5.0)

                step_loss, _ = criterion(pred_x, next_data_target.x, pred_p, dummy_target_p, node_weights=spatial_weights)
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
            
        if len(train_loader) > 0:
            avg_train_loss = total_train_loss / len(train_loader)
        else:
            avg_train_loss = 0.0
        
        loss_history["train_loss"].append(avg_train_loss)
        print(f"   Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}")
        
        # --- Validation Phase ---
        # (简单起见，这里只跑验证循环，不更新梯度)
        encoder.eval(); processor.eval(); decoder.eval()
        total_val_loss = 0
        
        # Diagnostic storage for this epoch (store data from the FIRST batch only efficiently)
        diag_times = []
        diag_pred_min = []
        diag_pred_max = []
        diag_truth_min = []
        diag_truth_max = []

        with torch.no_grad():
            for batch_idx, batch_sequence in enumerate(val_loader):
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
                
                # Iterate through the sequence time steps
                for t in range(SEQ_LEN - 1):
                    current_data = processed_sequence[t]
                    next_data_target = processed_sequence[t+1]
                    
                    latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)
                    next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
                    pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch, current_data.pos)
                    
                    dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)

                    if torch.isnan(pred_x).any() or torch.isnan(pred_p).any():
                         break
                    
                    # 计算空间权重 (Spatial Weighting)
                    spatial_weights = compute_spatial_weights(current_data.x[:, 0], current_data.batch, axis_weight=10.0, pedestal_weight=5.0)

                    step_loss, _ = criterion(pred_x, next_data_target.x, pred_p, dummy_target_p, node_weights=spatial_weights)
                    loss_accum += step_loss
                    
                    # --- Diagnostic Collection (First Batch Only) ---
                    if batch_idx == 0:
                        # Extract first graph from batch for clean consistency
                        # Note: pred_x is a batch of nodes. We need to slice the first graph's nodes.
                        # current_data.batch has node assignments.
                        
                        # Identify nodes belonging to graph 0
                        mask_graph_0 = (current_data.batch == 0)
                        
                        # Get Time (Target time t+1)
                        if hasattr(next_data_target, 'time'):
                             # If batched, 'time' is usually [Batch_Size]. Get first element.
                             t_val = next_data_target.time
                             if t_val.numel() > 1:
                                 t_val = t_val[0]
                             t_scalar = t_val.item()
                        else:
                             t_scalar = float(t + 1)

                        # Extract Psi (Channel 0) and denormalize (* 10.0)
                        # Ensure we select only the nodes for the first graph in the batch
                        mask_graph_0 = (current_data.batch == 0)
                        pred_psi_g0 = pred_x[mask_graph_0, 0] * 10.0
                        truth_psi_g0 = next_data_target.x[mask_graph_0, 0] * 10.0
                        
                        diag_times.append(t_scalar)
                        diag_pred_min.append(pred_psi_g0.min().item())
                        diag_pred_max.append(pred_psi_g0.max().item())
                        diag_truth_min.append(truth_psi_g0.min().item())
                        diag_truth_max.append(truth_psi_g0.max().item())
            
                total_val_loss += loss_accum.item() / (SEQ_LEN - 1)
        
        if len(val_loader) > 0:
            avg_val_loss = total_val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
            
        loss_history["val_loss"].append(avg_val_loss)
        print(f"   Epoch {epoch+1} Val Loss:   {avg_val_loss:.6f}")

        # --- Diagnostic Plotting (Every 5 Epochs or Final) ---
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            if diag_times:
                # Deduplicate and Sort
                # Since we loop sequence, t is increasing. But if we processed multiple batches (we don't for diag), it might be messy.
                # Here we only took batch_idx=0, so it is strictly one sequence.
                
                plt.figure(figsize=(10, 6))
                plt.plot(diag_times, diag_pred_min, 'r--', label='Pred Min')
                plt.plot(diag_times, diag_pred_max, 'r-', label='Pred Max')
                plt.plot(diag_times, diag_truth_min, 'b--', label='Truth Min', alpha=0.5)
                plt.plot(diag_times, diag_truth_max, 'b-', label='Truth Max', alpha=0.5)
                
                plt.xlabel("Time (s)")
                plt.ylabel("Psi (Poloidal Flux)")
                plt.title(f"Psi Evolution Diagnosis (Epoch {epoch+1})")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(VIS_DIR, f"psi_diag_epoch_{epoch+1}.png"))
                plt.close()

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

    # ================= 5. Plot Loss History =================
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["train_loss"], label="Train Loss")
    plt.plot(loss_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VIS_DIR, "loss_history.png"))
    plt.close()

    # ================= 6. 可视化测试结果 =================
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
        latent_x = encoder(input_data.x, input_data.edge_index, input_data.pos)
        next_latent_x, _ = processor(latent_x, input_data.batch) # 简化：无状态
        pred_x, _ = decoder(next_latent_x, input_data.edge_index, input_data.batch, input_data.pos)

    # 取 Batch 中的第一个图 (PyG Batch Supports indexing to retrieve single graph)
    # This automatically handles slicing of attributes like x, pos, and hopefully triangles if present
    first_graph = input_data[0]
    first_target_graph = target_data[0]
    
    # 获取真实物理时间
    # 注意: first_graph.time 是 input (t)，first_target_graph.time 是 target (t+dt)
    t_input = first_graph.time.item() if hasattr(first_graph, 'time') else t_inspect
    t_target = first_target_graph.time.item() if hasattr(first_target_graph, 'time') else t_inspect + 1
    
    pred_x_sample = pred_x[:first_graph.num_nodes, :]
    truth_x_sample = target_data[0].x
    pos_sample = first_graph.pos
    
    # Inverse Normalization for Visualization (if needed based on logic above)
    # Recall we scaled Psi by 0.1: new_x[:, 0] = new_x[:, 0] * 0.1
    # So we multiply by 10 to get back to physical units
    pred_val_psi = pred_x_sample[:, 0].cpu() * 10.0
    truth_val_psi = truth_x_sample[:, 0].cpu() * 10.0

    print(f"\n[Stats] Ground Truth Psi at t={t_target:.5f}: Min={truth_val_psi.min():.4f}, Max={truth_val_psi.max():.4f}")
    print(f"[Stats] Predicted Psi at t={t_target:.5f}:    Min={pred_val_psi.min():.4f}, Max={pred_val_psi.max():.4f}")
    
    # Extract Triangles for Visualization
    vis_triangles = None
    if hasattr(first_graph, 'triangles') and first_graph.triangles is not None:
        vis_triangles = first_graph.triangles
        print(f"   Using explicit mesh topology ({vis_triangles.shape[0]} triangles) for visualization.")
    else:
        print("   Warning: 'triangles' attribute not found in batch. Using Delaunay triangulation.")

    # 绘制 Flux (psi) - Channel 0
    try:
        plot_jorek_field(
            pos=pos_sample.cpu(), 
            values=pred_val_psi,
            triangles=vis_triangles, 
            title=f"Predicted Psi (t={t_target:.5f})", 
            save_path=os.path.join(VIS_DIR, "pred_psi.png")
        )
        
        plot_jorek_field(
            pos=pos_sample.cpu(), 
            values=truth_val_psi, 
            triangles=vis_triangles,
            title=f"Ground Truth Psi (t={t_target:.5f})", 
            save_path=os.path.join(VIS_DIR, "truth_psi.png")
        )
        print(f"✅ Training Pipeline Completed! Results in {OUTPUT_DIR}")
    except ValueError as e:
        print(f"⚠️ Visualization failed due to NaN/Inf in predictions: {e}")
        print("This indicates model divergence. Please adjust hyperparameters (LR, Norm) or check data.")
        print(f"Model checkpoints saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    train_sequence_pipeline()
