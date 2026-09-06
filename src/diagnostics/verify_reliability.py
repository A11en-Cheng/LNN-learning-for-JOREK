import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import JorekSequenceDataset, collate_temporal_graphs
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder
from src.data.model502_schema import FIELD_INDEX, FIELD_NAMES, inverse_transform_state, transform_state

def verify_reliability():
    # ================= 配置参数 =================
    DATA_DIR = "data/EHL-2/scan_1d22_ang12_massloss_neon"
    OUTPUT_DIR = "output/verification_model502"
    CHECKPOINT_PATH = "output/model502_baseline/checkpoints/latest_model.pt"
    
    WARMUP_STEPS = 6
    ROLLOUT_STEPS = 40
    SEQ_LEN = WARMUP_STEPS + ROLLOUT_STEPS
    
    HIDDEN_DIM = 64
    LATENT_DIM = 32
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*50)
    print("🔍 HyPhy-GLNN Reliability & Physical Fidelity Verification")
    print("="*50)

    # ================= 1. 数据集准备与全局归一化 =================
    print(f"\n[1/4] Loading Dataset (Sequence Length: {SEQ_LEN})...")
    dataset = JorekSequenceDataset(root_dir=DATA_DIR, seq_len=SEQ_LEN)
    
    if len(dataset) == 0:
        print(f"Error: Not enough data for a sequence of length {SEQ_LEN}.")
        return
        
    # 取第一个长序列进行测试
    test_seq = dataset[0] 
    
    print("Computing Global Normalization Statistics...")
    # Diagnostic-only statistics. Production checkpoint statistics must come
    # from the training split manifest.
    all_x = transform_state(torch.cat([data.x for data in test_seq], dim=0))
    global_mean = all_x.mean(dim=0)
    global_std = all_x.std(dim=0) + 1e-6

    # ================= 2. 模型构建与权重加载 =================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2/4] Building Models on {device}...")
    
    global_mean = global_mean.to(device)
    global_std = global_std.to(device)
    
    encoder = JorekGNNEncoder(input_dim=9, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=4).to(device)
    processor = JorekLNNProcessor(latent_dim=LATENT_DIM, hidden_size=HIDDEN_DIM, num_layers=2).to(device)
    decoder = JorekGNNDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=9, num_layers=4).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        processor.load_state_dict(checkpoint['processor'])
        decoder.load_state_dict(checkpoint['decoder'])

        # 如果 checkpoint 中包含训练时的全局统计量，则优先生效
        if 'global_mean' in checkpoint and 'global_std' in checkpoint:
            print("Using global_mean/std from checkpoint for consistency with training...")
            global_mean = checkpoint['global_mean'].to(device)
            global_std = checkpoint['global_std'].to(device)
    else:
        print("⚠️ WARNING: No checkpoint found. Running with untrained initialized weights for pipeline verification.")

    encoder.eval(); processor.eval(); decoder.eval()

    # ================= 3. 自回归 Rollout 测试 (Autoregressive Rollout) =================
    print(f"\n[3/4] Performing Autoregressive Rollout ({WARMUP_STEPS} Warmup + {ROLLOUT_STEPS} Free Run)...")
    
    hidden_state = None
    predictions_real = []
    ground_truths_real = []
    times = []
    
    with torch.no_grad():
        # --- Phase 1: Warm-up (Teacher Forcing) ---
        for t in range(WARMUP_STEPS):
            gt_data = test_seq[t].to(device)

            # Same canonical transform used by Model 502 training.
            new_x = transform_state(gt_data.x)
            norm_x = (new_x - global_mean) / global_std
            
            latent = encoder(norm_x, gt_data.edge_index, gt_data.pos)
            latent, hidden_state = processor(latent, getattr(gt_data, 'batch', None), hidden_state)

            # Phase 2.5: 解码器输出 Δx（归一化空间残差）
            pred_delta_norm, _ = decoder(
                latent,
                gt_data.edge_index,
                getattr(gt_data, 'batch', None),
                gt_data.pos,
            )

            # 得到该步对下一时刻的预测：x_{t+1} = x_t + Δx_t
            pred_next_norm = norm_x + pred_delta_norm
            
            if t == WARMUP_STEPS - 1:
                # 用最后一个 warm-up 步的预测作为 Free Run 初始状态
                current_pred_norm = pred_next_norm
                
        # --- Phase 2: Free Run (Autoregressive, Residual Form) ---
        for t in range(WARMUP_STEPS, SEQ_LEN):
            gt_data = test_seq[t].to(device)
            
            # Reverse Z-score and canonical log-space transform.
            pred_real = inverse_transform_state(current_pred_norm * global_std + global_mean)
            gt_raw = gt_data.x.clone()

            gt_real = gt_raw
            
            predictions_real.append(pred_real.cpu())
            ground_truths_real.append(gt_real.cpu())
            times.append(t)
            
            if t < SEQ_LEN - 1:
                # 关键：将上一时间步的预测值作为当前时间步的输入（残差形式）
                # current_pred_norm 已经是 log+Z-score 之后的归一化状态，直接送入网络
                latent = encoder(current_pred_norm, gt_data.edge_index, gt_data.pos)
                latent, hidden_state = processor(latent, getattr(gt_data, 'batch', None), hidden_state)

                pred_delta_norm, _ = decoder(
                    latent,
                    gt_data.edge_index,
                    getattr(gt_data, 'batch', None),
                    gt_data.pos,
                )

                # 自回归：x_{t+1} = x_t + Δx_t（全程在归一化空间中演化）
                current_pred_norm = current_pred_norm + pred_delta_norm

    # 计算误差累积 (Error Accumulation)
    error_accumulation = {name: [] for name in FIELD_NAMES}
    var_keys = list(error_accumulation.keys())
    
    for p, gt in zip(predictions_real, ground_truths_real):
        # 计算每个变量的相对 MAE (Mean Absolute Error)
        for i, key in enumerate(var_keys):
            mae = torch.abs(p[:, i] - gt[:, i]).mean().item()
            # 使用全局标准差作为分母进行相对误差缩放，避免量级影响
            rel_error = mae / (global_std[i].item() + 1e-8)
            error_accumulation[key].append(rel_error)

    # 绘制误差累积图
    plt.figure(figsize=(10, 6))
    for key in var_keys:
        plt.plot(times, error_accumulation[key], label=key, marker='o', markersize=3)
    plt.axvline(x=WARMUP_STEPS, color='k', linestyle='--', label='Start Free Run')
    plt.title("Autoregressive Rollout Error Accumulation (Relative MAE)")
    plt.xlabel("Time Step")
    plt.ylabel("Relative Error (MAE / Global Std)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "autoregressive_error_accumulation.png"))
    plt.close()
    print("   -> Saved autoregressive_error_accumulation.png")

    # ================= 4. 中平面剖面分析 (Midplane Profile Analysis) =================
    print("\n[4/4] Analyzing Midplane Profiles (Pedestal Gradients)...")
    
    # 取 Rollout 的最后一步进行分析
    last_idx = -1
    pred_last = predictions_real[last_idx]
    gt_last = ground_truths_real[last_idx]
    pos = test_seq[-1].pos.cpu()
    
    R_coords = pos[:, 0]
    Z_coords = pos[:, 1]
    
    # 提取 Z 接近 0 的节点 (Midplane)
    Z_TOLERANCE = 0.05
    midplane_mask = torch.abs(Z_coords) < Z_TOLERANCE
    
    R_mid = R_coords[midplane_mask]
    
    # 提取 Rho 和 Te
    pred_rho_mid = pred_last[midplane_mask, 2]
    gt_rho_mid = gt_last[midplane_mask, 2]
    
    pred_Te_mid = pred_last[midplane_mask, 3]
    gt_Te_mid = gt_last[midplane_mask, 3]
    
    # 按 R 坐标排序以便绘图
    sort_indices = torch.argsort(R_mid)
    R_mid_sorted = R_mid[sort_indices].numpy()
    
    pred_rho_sorted = pred_rho_mid[sort_indices].numpy()
    gt_rho_sorted = gt_rho_mid[sort_indices].numpy()
    
    pred_Te_sorted = pred_Te_mid[sort_indices].numpy()
    gt_Te_sorted = gt_Te_mid[sort_indices].numpy()
    
    # 绘制 1D 剖面图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rho Profile
    ax1.plot(R_mid_sorted, gt_rho_sorted, 'b-', label='Ground Truth', linewidth=2)
    ax1.plot(R_mid_sorted, pred_rho_sorted, 'r--', label='Prediction (Rollout)', linewidth=2)
    ax1.set_title(f"Midplane Density (Rho) Profile at Step {SEQ_LEN-1}")
    ax1.set_xlabel("Major Radius R (m)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True)
    
    # Te Profile
    ax2.plot(R_mid_sorted, gt_Te_sorted, 'b-', label='Ground Truth', linewidth=2)
    ax2.plot(R_mid_sorted, pred_Te_sorted, 'r--', label='Prediction (Rollout)', linewidth=2)
    ax2.set_title(f"Midplane Temperature (Te) Profile at Step {SEQ_LEN-1}")
    ax2.set_xlabel("Major Radius R (m)")
    ax2.set_ylabel("Temperature")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "midplane_profiles_pedestal.png"))
    plt.close()
    print("   -> Saved midplane_profiles_pedestal.png")
    
    print(f"\n✅ Verification complete. All diagnostics saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    verify_reliability()
