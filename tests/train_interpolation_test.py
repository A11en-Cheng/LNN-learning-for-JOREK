import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import JorekSequenceDataset, collate_temporal_graphs
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder
from src.training.loss import JorekLoss
from src.models.loss import compute_spatial_weights
from src.utils.visualization import plot_jorek_field

def train_interpolation_test():
    # ================= 配置参数 =================
    DATA_DIR = "data/test_data_for_pipeline_EHL-2"
    OUTPUT_DIR = "output/interpolation_test"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    VIS_DIR = os.path.join(OUTPUT_DIR, "vis")
    
    BATCH_SIZE = 4
    SEQ_LEN = 10        # 序列长度
    EPOCHS = 3000        # 训练轮次
    LR = 5e-4
    
    # 模型规模参数
    HIDDEN_DIM = 64
    LATENT_DIM = 32
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"==================================================")
    print(f"🚀 Starting Interpolation Test (Held-out Test Set)")
    print(f"   Batch: {BATCH_SIZE}, Seq Len: {SEQ_LEN}, Epochs: {EPOCHS}")
    print(f"==================================================")

    # ================= 1. 数据集准备与插值切分 =================
    print("\n[1/5] Initializing Dataset & Splitting by Interpolation Rule...")
    full_dataset = JorekSequenceDataset(root_dir=DATA_DIR, seq_len=SEQ_LEN)
    
    if len(full_dataset) == 0:
        print("Error: Dataset is empty.")
        return

    # 策略：每 5 个样本保留第 5 个作为测试集 (Indices: 4, 9, 14...)
    all_indices = list(range(len(full_dataset)))
    check_interval = 5
    
    train_indices = [i for i in all_indices if (i + 1) % check_interval != 0]
    test_indices = [i for i in all_indices if (i + 1) % check_interval == 0]
    
    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    print(f"   Total Samples: {len(full_dataset)}")
    print(f"   Train Samples: {len(train_subset)} (Indices like 0,1,2,3, 5,6,7,8...)")
    print(f"   Test Samples:  {len(test_subset)}  (Indices like 4, 9, 14...)")
    
    num_workers = min(4, os.cpu_count() or 1) 
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_temporal_graphs, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, collate_fn=collate_temporal_graphs, num_workers=num_workers)

    # ================= 2. 全局归一化统计 (Rule 1) =================
    print("\n[2/5] Computing Global Normalization Statistics...")
    # 为了避免加载整个数据集，我们从 train_loader 中抽取部分 batch 计算全局均值和标准差
    all_x_samples = []
    for i, batch_sequence in enumerate(train_loader):
        for data in batch_sequence:
            all_x_samples.append(data.x)
        if i >= 5: # 取前 6 个 batch 估算即可
            break
    all_x_tensor = torch.cat(all_x_samples, dim=0)
    
    # Phase 1.5: Log-Space Mapping for Thermodynamic Variables (rho, Te)
    # Apply log1p(ReLU(x)) to compress dynamic range before Z-score
    all_x_tensor[:, 2] = torch.log1p(torch.relu(all_x_tensor[:, 2]))
    all_x_tensor[:, 3] = torch.log1p(torch.relu(all_x_tensor[:, 3]))
    
    global_mean = all_x_tensor.mean(dim=0)
    global_std = all_x_tensor.std(dim=0) + 1e-6
    print(f"   Global Mean: {global_mean.tolist()}")
    print(f"   Global Std:  {global_std.tolist()}")

    # ================= 3. 模型构建 =================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3/5] Building Models on device: {device}")
    
    global_mean = global_mean.to(device)
    global_std = global_std.to(device)
    
    encoder = JorekGNNEncoder(input_dim=5, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=4).to(device)
    processor = JorekLNNProcessor(latent_dim=LATENT_DIM, hidden_size=HIDDEN_DIM, num_layers=2).to(device)
    decoder = JorekGNNDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=5, num_layers=4).to(device)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(processor.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=1e-5
    )
    
    # 赋予不同物理场权重，防止小量级场被忽略 (Rule 3)
    # 传入 stats 用于物理损失中的反归一化
    stats = {'mean': global_mean, 'std': global_std}
    criterion = JorekLoss(lambda_data=1.0, lambda_phys=0.0, stats=stats, dt=1.0)

    loss_history = {"train_loss": [], "test_loss": []}
    # 记录各个物理场的独立 Loss
    channel_loss_history = {
        "loss_psi": [], "loss_phi": [], "loss_rho": [], "loss_Te": [], "loss_vpar": []
    }
    var_names = ["Psi (Flux)", "Phi (Stream)", "Rho (Density)", "T (Temp)", "v_par (Velocity)"]

    # ================= 4. 训练与诊断循环 =================
    print("\n[4/5] Starting Training Loop...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- Physics Loss Annealing ---
        if epoch <= 1000:
            criterion.lambda_phys = 0.0
        elif epoch <= 2500:
            # Linearly increase from 0.0 to 0.005
            criterion.lambda_phys = 0.005 * (epoch - 1000) / 1500.0
        else:
            criterion.lambda_phys = 0.005
            
        # --- Training ---
        encoder.train(); processor.train(); decoder.train()
        total_train_loss = 0
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch}/{EPOCHS} running... (lambda_phys: {criterion.lambda_phys:.5f})")

        for batch_sequence in train_loader:
            optimizer.zero_grad()
            loss_accum = 0
            hidden_state = None # 序列起始，重置状态 (Rule 4)
            
            # 累加每个 batch 的通道 loss
            batch_channel_losses = {k: 0.0 for k in channel_loss_history.keys()}
            
            # Pre-processing: Global Normalization
            processed_sequence = []
            for data in batch_sequence:
                d = data.to(device)
                if hasattr(d, 'x'):
                    new_x = d.x.clone()
                    # Phase 1.5: Log-Space Mapping for Thermodynamic Variables
                    new_x[:, 2] = torch.log1p(torch.relu(new_x[:, 2]))
                    new_x[:, 3] = torch.log1p(torch.relu(new_x[:, 3]))
                    # 严格执行全局 Z-score 归一化，不使用 arbitrary constants
                    d.x = (new_x - global_mean) / global_std
                processed_sequence.append(d)

            # Sequence Loop (Unroll temporal sequence)
            for t in range(SEQ_LEN - 1):
                current_data = processed_sequence[t]
                next_data_target = processed_sequence[t+1]
                
                latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)
                next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
                pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch, current_data.pos)
                
                dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)
                
                if torch.isnan(pred_x).any():
                    loss_accum = torch.tensor(0.0, device=device, requires_grad=True)
                    break

                # 计算空间权重 (Spatial Weighting)
                # 使用 current_data.x[:, 0] (Psi) 来计算权重
                spatial_weights = compute_spatial_weights(current_data.x[:, 0], current_data.batch, axis_weight=10.0, pedestal_weight=5.0)
                
                # JorekLoss 现在接受 pred, target, graph_data, curr_state
                step_loss = criterion(pred_x, next_data_target.x, current_data, curr_state=current_data.x)
                loss_accum += step_loss
                
                # 累加通道 loss (这里简单记录 MSE 作为 diagnostics)
                with torch.no_grad():
                    mse_per_channel = torch.mean((pred_x - next_data_target.x)**2, dim=0)
                    batch_channel_losses["loss_psi"] += mse_per_channel[0].item()
                    batch_channel_losses["loss_phi"] += mse_per_channel[1].item()
                    batch_channel_losses["loss_rho"] += mse_per_channel[2].item()
                    batch_channel_losses["loss_Te"] += mse_per_channel[3].item()
                    batch_channel_losses["loss_vpar"] += mse_per_channel[4].item()
            
            if loss_accum.item() == 0.0: continue

            avg_seq_loss = loss_accum / (SEQ_LEN - 1)
            avg_seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_train_loss += avg_seq_loss.item()
            
            # 记录平均通道 loss
            for k in batch_channel_losses.keys():
                channel_loss_history[k].append(batch_channel_losses[k] / (SEQ_LEN - 1))
            
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        loss_history["train_loss"].append(avg_train_loss)
        
        # --- Testing & Real-time Diagnostics ---
        encoder.eval(); processor.eval(); decoder.eval()
        total_test_loss = 0
        
        # 仅在需要诊断的 Epoch 收集详细统计信息
        do_diagnostics = (epoch % 5 == 0 or epoch == EPOCHS)
        diag_data = {v: {"time": [], "pred_min": [], "pred_max": [], "truth_min": [], "truth_max": []} for v in var_names}
        
        with torch.no_grad():
            for i, batch_sequence in enumerate(test_loader):
                processed_sequence = []
                for data in batch_sequence:
                    d = data.to(device)
                    if hasattr(d, 'x'):
                        new_x = d.x.clone()
                        # Phase 1.5: Log-Space Mapping for Thermodynamic Variables
                        new_x[:, 2] = torch.log1p(torch.relu(new_x[:, 2]))
                        new_x[:, 3] = torch.log1p(torch.relu(new_x[:, 3]))
                        d.x = (new_x - global_mean) / global_std
                    processed_sequence.append(d)

                hidden_state = None
                loss_accum = 0
                
                for t in range(SEQ_LEN - 1):
                    current_data = processed_sequence[t]
                    next_data_target = processed_sequence[t+1]
                    
                    latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)
                    next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
                    pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch, current_data.pos)
                    
                    dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)
                    
                    # 计算空间权重 (Spatial Weighting)
                    spatial_weights = compute_spatial_weights(current_data.x[:, 0], current_data.batch, axis_weight=10.0, pedestal_weight=5.0)
                    
                    loss_s = criterion(pred_x, next_data_target.x, current_data, curr_state=current_data.x)
                    loss_accum += loss_s
                    
                    # --- 实时诊断逻辑：提取第一个 Batch 的第一条序列 ---
                    if do_diagnostics and i == 0:
                        # 获取物理时间戳 t_now
                        if hasattr(next_data_target, 'time'):
                            t_val = next_data_target.time
                            t_scalar = t_val[0].item() if t_val.numel() > 1 else t_val.item()
                        else:
                            t_scalar = float(t)
                        
                        # 仅提取第一张图 (batch == 0)
                        mask = (current_data.batch == 0)
                        
                        # 反归一化以获取真实的物理量级
                        pred_x_real = pred_x[mask] * global_std + global_mean
                        truth_x_real = next_data_target.x[mask] * global_std + global_mean
                        
                        # Phase 1.5: Reverse Log-Space Mapping for Thermodynamic Variables
                        pred_x_real[:, 2] = torch.expm1(pred_x_real[:, 2])
                        pred_x_real[:, 3] = torch.expm1(pred_x_real[:, 3])
                        truth_x_real[:, 2] = torch.expm1(truth_x_real[:, 2])
                        truth_x_real[:, 3] = torch.expm1(truth_x_real[:, 3])
                        
                        for ch_idx, var_name in enumerate(var_names):
                            p_val = pred_x_real[:, ch_idx]
                            t_val_real = truth_x_real[:, ch_idx]
                            
                            diag_data[var_name]["time"].append(t_scalar)
                            diag_data[var_name]["pred_min"].append(p_val.min().item())
                            diag_data[var_name]["pred_max"].append(p_val.max().item())
                            diag_data[var_name]["truth_min"].append(t_val_real.min().item())
                            diag_data[var_name]["truth_max"].append(t_val_real.max().item())

                total_test_loss += (loss_accum / (SEQ_LEN - 1)).item()
                
        avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        loss_history["test_loss"].append(avg_test_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

        # ================= 保存模型权重 =================
        # 每 10 个 Epoch 保存一次，或者在最后一个 Epoch 保存
        if epoch % 10 == 0 or epoch == EPOCHS:
            checkpoint_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'processor': processor.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_mean': global_mean,
                'global_std': global_std
            }, checkpoint_path)
            
            # 始终保存一个 latest_model.pt 供验证脚本使用
            latest_path = os.path.join(MODEL_DIR, "latest_model.pt")
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'processor': processor.state_dict(),
                'decoder': decoder.state_dict(),
                'global_mean': global_mean,
                'global_std': global_std
            }, latest_path)

        # ================= 实时诊断绘图 =================
        if do_diagnostics:
            # 1. 生成 Psi (Poloidal Flux) 的 Min/Max 趋势对比图
            times = diag_data["Psi (Flux)"]["time"]
            plt.figure(figsize=(8, 5))
            plt.plot(times, diag_data["Psi (Flux)"]["pred_min"], 'r--', label='Pred Min')
            plt.plot(times, diag_data["Psi (Flux)"]["pred_max"], 'r-', label='Pred Max')
            plt.plot(times, diag_data["Psi (Flux)"]["truth_min"], 'b--', label='Truth Min')
            plt.plot(times, diag_data["Psi (Flux)"]["truth_max"], 'b-', label='Truth Max')
            plt.title(f"Psi Min/Max Trend (Epoch {epoch})")
            plt.xlabel("Simulation Time (t_now)")
            plt.ylabel("Psi Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(VIS_DIR, f"psi_diag_epoch_{epoch}.png"))
            plt.close()
            
            # 2. 可视化测试集的场和预测的场 (同图对比)
            # 使用最后一步的数据
            pos = current_data.pos[mask].cpu()
            tris = current_data.triangles.cpu() if hasattr(current_data, 'triangles') else None
            
            true_psi = truth_x_real[:, 0].cpu().numpy()
            pred_psi = pred_x_real[:, 0].cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 如果有三角形连接信息，使用 tricontourf，否则使用 scatter
            if tris is not None:
                tc1 = ax1.tricontourf(pos[:, 0].numpy(), pos[:, 1].numpy(), tris.numpy(), true_psi, levels=50, cmap='jet')
                tc2 = ax2.tricontourf(pos[:, 0].numpy(), pos[:, 1].numpy(), tris.numpy(), pred_psi, levels=50, cmap='jet')
            else:
                tc1 = ax1.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), c=true_psi, cmap='jet', s=1)
                tc2 = ax2.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), c=pred_psi, cmap='jet', s=1)
                
            ax1.set_title(f"True Psi at t={times[-1]:.5f}")
            fig.colorbar(tc1, ax=ax1)
            ax1.set_aspect('equal', adjustable='box')
            ax2.set_title(f"Predicted Psi at t={times[-1]:.5f}")
            fig.colorbar(tc2, ax=ax2)
            ax2.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, f"field_comparison_epoch_{epoch}.png"))
            plt.close()

    # ================= 5. 最终统计结果可视化 =================
    print("\n[5/5] Generating Final Statistical Visualization...")
    
    # 绘制 5 个变量的最终趋势图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, var_name in enumerate(var_names):
        ax = axes[idx]
        d = diag_data[var_name]
        times = d["time"]
        
        ax.plot(times, d["pred_min"], 'r--', label='Pred Min')
        ax.plot(times, d["pred_max"], 'r-', label='Pred Max')
        ax.plot(times, d["truth_min"], 'b--', label='Truth Min', alpha=0.5)
        ax.plot(times, d["truth_max"], 'b-', label='Truth Max', alpha=0.5)
        
        ax.set_title(f"{var_name} Evolution")
        ax.set_xlabel("Simulation Time (t_now)")
        ax.grid(True)
        if idx == 0: ax.legend()

    if len(var_names) < 6:
        fig.delaxes(axes[5])
        
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "final_variables_stats_evolution.png"))
    plt.close()

    # Loss Plot
    plt.figure()
    plt.plot(loss_history["train_loss"], label="Train")
    plt.plot(loss_history["test_loss"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(VIS_DIR, "loss_history.png"))
    plt.close()

    # 绘制各个物理场的独立 Loss 曲线
    plt.figure(figsize=(10, 6))
    for k, v in channel_loss_history.items():
        # 由于记录的是每个 batch 的 loss，这里可以平滑一下或者直接画
        # 为了清晰，我们计算每个 epoch 的平均 loss
        batches_per_epoch = len(train_loader)
        if batches_per_epoch > 0:
            epoch_avg_losses = [np.mean(v[i*batches_per_epoch:(i+1)*batches_per_epoch]) for i in range(EPOCHS)]
            plt.plot(epoch_avg_losses, label=k)
    
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("Channel-wise Loss Evolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VIS_DIR, "channel_loss_history.png"))
    plt.close()

    print(f"✅ Interpolation Test Completed. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    train_interpolation_test()
