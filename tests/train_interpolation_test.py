import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import JorekSequenceDataset, collate_temporal_graphs
from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.loss import (
    compute_spatial_weights,
    compute_normalized_flux,
    JorekPhysicsLoss,
)
from src.utils.visualization import plot_jorek_field

def load_config(config_path="config.yaml"):
    """加载 YAML 配置，强制使用 UTF-8，避免 Windows 默认 GBK 造成解码错误。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return yaml.safe_load(text)

def train_interpolation_test(config_path="config.yaml"):
    # ================= 配置参数 =================
    config = load_config(config_path)
    
    DATA_DIR = config['data']['data_dir']
    OUTPUT_DIR = config['data']['output_dir']
    MODEL_DIR = config['data']['model_dir']
    VIS_DIR = config['data']['vis_dir']
    
    BATCH_SIZE = config['training']['batch_size']
    SEQ_LEN = config['training']['seq_len']
    EPOCHS = config['training']['epochs']
    LR = config['training']['lr']
    WEIGHT_DECAY = config['training']['weight_decay']
    NUM_WORKERS = config['training']['num_workers']
    
    # 模型规模参数
    INPUT_DIM = config['model']['input_dim']
    HIDDEN_DIM = config['model']['hidden_dim']
    LATENT_DIM = config['model']['latent_dim']
    ENC_LAYERS = config['model']['encoder_layers']
    PROC_LAYERS = config['model']['processor_layers']
    DEC_LAYERS = config['model']['decoder_layers']
    
    # 物理损失参数
    LAMBDA_DATA = config['physics']['lambda_data']
    LAMBDA_PHYS_MAX = config['physics']['lambda_phys_max']
    # 物理损失退火调度参数（Phase 2.6）
    #   Epoch 0–PHYS_WARMUP:      lambda_phys = 0
    #   Epoch PHYS_WARMUP–PHYS_ANNEAL: 从 0 线性增加到 LAMBDA_PHYS_MAX
    #   Epoch > PHYS_ANNEAL:      固定为 LAMBDA_PHYS_MAX
    PHYS_WARMUP = config['physics']['phys_warmup_epochs']
    PHYS_ANNEAL = config['physics']['phys_anneal_epochs']
    DT = config['physics']['dt']
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"==================================================")
    print(f"🚀 Starting Interpolation Test (Held-out Test Set)")
    print(f"   Config: {config_path}")
    print(f"   Batch: {BATCH_SIZE}, Seq Len: {SEQ_LEN}, Epochs: {EPOCHS}")
    print(f"==================================================")

    # ================= 1. 数据集准备与插值切分 =================
    print("\n[1/5] Initializing Dataset & Random Train/Test Split...")
    full_dataset = JorekSequenceDataset(root_dir=DATA_DIR, seq_len=SEQ_LEN)

    # 计算整个模拟的全局时间范围，用于诊断图的时间坐标（psi_diag_epoch）
    search_path = os.path.join(DATA_DIR, "jorek*.h5")
    h5_files = glob.glob(search_path)
    def _extract_num(fp):
        m = re.search(r"(\d+)", os.path.basename(fp))
        return int(m.group(1)) if m else 0
    h5_files_sorted = sorted(h5_files, key=_extract_num)
    if len(h5_files_sorted) >= 2:
        t_parser_start = JorekDataParser(h5_files_sorted[0])
        t_start_graph = t_parser_start.parse_to_graph()
        GLOBAL_T_MIN = float(t_start_graph.time.view(-1)[0].item())
        t_parser_end = JorekDataParser(h5_files_sorted[-1])
        t_end_graph = t_parser_end.parse_to_graph()
        GLOBAL_T_MAX = float(t_end_graph.time.view(-1)[0].item())
    else:
        GLOBAL_T_MIN = 0.0
        GLOBAL_T_MAX = 1.0
    
    if len(full_dataset) == 0:
        print("Error: Dataset is empty.")
        return

    # 策略更新：使用随机划分训练 / 测试集，而不是固定间隔采样
    # 这样可以避免时间上均匀抽样带来的“插值偏置”，更像通用时序预测任务
    dataset_size = len(full_dataset)
    all_indices = np.arange(dataset_size)
    # 为了可复现，固定一个随机种子；如需外部控制，可改为从 config 读取
    np.random.seed(42)
    np.random.shuffle(all_indices)

    test_ratio = 0.2
    split_idx = int(dataset_size * (1.0 - test_ratio))
    train_indices = all_indices[:split_idx].tolist()
    test_indices = all_indices[split_idx:].tolist()
    
    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    print(f"   Total Samples: {len(full_dataset)}")
    print(f"   Train Samples: {len(train_subset)} (random indices)")
    print(f"   Test Samples:  {len(test_subset)}  (random indices)")
    
    num_workers = min(NUM_WORKERS, os.cpu_count() or 1) 
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
    
    encoder = JorekGNNEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=ENC_LAYERS).to(device)
    processor = JorekLNNProcessor(latent_dim=LATENT_DIM, hidden_size=HIDDEN_DIM, num_layers=PROC_LAYERS).to(device)
    decoder = JorekGNNDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM, num_layers=DEC_LAYERS).to(device)

    # Load pretrained models if specified
    # 支持两种形式：
    # 1) 直接是某个模块的 state_dict 文件
    # 2) 组合 checkpoint（例如 latest_model.pt），内部包含 'encoder'/'processor'/'decoder' 键
    def _load_maybe_composite(path, key):
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
        return ckpt

    if config['pretrained']['encoder']:
        enc_path = config['pretrained']['encoder']
        print(f"   Loading pretrained encoder from {enc_path}")
        encoder.load_state_dict(_load_maybe_composite(enc_path, 'encoder'))
    if config['pretrained']['processor']:
        proc_path = config['pretrained']['processor']
        print(f"   Loading pretrained processor from {proc_path}")
        processor.load_state_dict(_load_maybe_composite(proc_path, 'processor'))
    if config['pretrained']['decoder']:
        dec_path = config['pretrained']['decoder']
        print(f"   Loading pretrained decoder from {dec_path}")
        decoder.load_state_dict(_load_maybe_composite(dec_path, 'decoder'))
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(processor.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    
    # 赋予不同物理场权重，防止小量级场被忽略 (Rule 3)
    # 传入 stats 用于物理损失中的反归一化
    stats = {'mean': global_mean, 'std': global_std}
    # Phase 2.6: 使用 Physics-Informed Loss（退火方式注入），lambda_phys 在训练循环中动态更新
    criterion = JorekPhysicsLoss(lambda_data=LAMBDA_DATA, lambda_phys=0.0, stats=stats, dt=DT)

    loss_history = {"train_loss": [], "test_loss": []}
    # 记录各个物理场的独立 Loss
    channel_loss_history = {
        "loss_psi": [], "loss_phi": [], "loss_rho": [], "loss_Te": [], "loss_vpar": []
    }
    var_names = ["Psi (Flux)", "Phi (Stream)", "Rho (Density)", "T (Temp)", "v_par (Velocity)"]

    # ================= 4. 训练与诊断循环 =================
    print("\n[4/5] Starting Training Loop...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- Physics Loss with Annealing Schedule (Phase 2.6) ---
        #  Epoch 0–PHYS_WARMUP:                lambda_phys = 0.0
        #  Epoch PHYS_WARMUP–PHYS_ANNEAL:      线性从 0 增加到 LAMBDA_PHYS_MAX
        #  Epoch > PHYS_ANNEAL:                固定为 LAMBDA_PHYS_MAX
        if LAMBDA_PHYS_MAX <= 0.0:
            criterion.lambda_phys = 0.0
        else:
            if epoch <= PHYS_WARMUP:
                lambda_phys_epoch = 0.0
            elif epoch <= PHYS_ANNEAL and PHYS_ANNEAL > PHYS_WARMUP:
                frac = float(epoch - PHYS_WARMUP) / float(PHYS_ANNEAL - PHYS_WARMUP)
                frac = max(0.0, min(1.0, frac))
                lambda_phys_epoch = LAMBDA_PHYS_MAX * frac
            else:
                lambda_phys_epoch = LAMBDA_PHYS_MAX

            criterion.lambda_phys = lambda_phys_epoch
        
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

                # 利用 Data.time 构造当前时间步的物理时间间隔 Δt（标量），并传入 LNN
                dt_scalar = None
                if hasattr(current_data, 'time') and hasattr(next_data_target, 'time'):
                    t_curr = current_data.time.to(device).view(-1)
                    t_next = next_data_target.time.to(device).view(-1)
                    dt_vals = t_next - t_curr
                    dt_scalar = dt_vals.mean()  # 对 batch 取平均，作为本时间步的代表性 Δt

                next_latent_x, hidden_state = processor(
                    latent_x,
                    current_data.batch,
                    state=hidden_state,
                    time_steps=dt_scalar,
                )

                # Phase 2.5: Decoder 预测的是 Δx（归一化空间的残差）
                pred_delta_x, pred_p = decoder(
                    next_latent_x,
                    current_data.edge_index,
                    current_data.batch,
                    current_data.pos,
                )

                # 将残差加到当前归一化状态上，得到对下一步状态的预测
                pred_x_norm = current_data.x + pred_delta_x
                
                dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)

                # 防止数值崩溃：如果预测中出现 NaN，则丢弃该 batch
                if torch.isnan(pred_x_norm).any() or torch.isnan(pred_p).any():
                    loss_accum = torch.tensor(0.0, device=device, requires_grad=True)
                    break

                # # 计算空间权重 (Spatial Weighting)
                # # 使用 current_data.x[:, 0] (Psi) 来计算权重
                # 目前弃用，因为 JorekPhysicsLoss 内部已经实现了基于当前状态的物理损失加权机制
                # spatial_weights = compute_spatial_weights(current_data.x[:, 0], current_data.batch, axis_weight=10.0, pedestal_weight=5.0)
                
                # JorekLoss 现在接受 pred, target, graph_data, curr_state
                boundary_attr = current_data.boundary if hasattr(current_data, 'boundary') else None
                freeboundary_attr = current_data.freeboundary if hasattr(current_data, 'freeboundary') else None
                step_loss = criterion(
                    pred_x_norm,
                    next_data_target.x,
                    current_data,
                    curr_state=current_data.x,
                    boundary=boundary_attr,
                    freeboundary=freeboundary_attr,
                )
                loss_accum += step_loss
                
                # 累加通道 loss (这里简单记录 MSE 作为 diagnostics)
                with torch.no_grad():
                    mse_per_channel = torch.mean((pred_x_norm - next_data_target.x)**2, dim=0)
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

                    # 同样在测试阶段向 LNN 提供当前时间步的物理时间间隔 Δt（标量）
                    dt_scalar = None
                    if hasattr(current_data, 'time') and hasattr(next_data_target, 'time'):
                        t_curr = current_data.time.to(device).view(-1)
                        t_next = next_data_target.time.to(device).view(-1)
                        dt_vals = t_next - t_curr
                        dt_scalar = dt_vals.mean()

                    next_latent_x, hidden_state = processor(
                        latent_x,
                        current_data.batch,
                        state=hidden_state,
                        time_steps=dt_scalar,
                    )

                    # Phase 2.5: 测试阶段同样使用 Δx 残差预测
                    pred_delta_x, pred_p = decoder(
                        next_latent_x,
                        current_data.edge_index,
                        current_data.batch,
                        current_data.pos,
                    )

                    # 累加残差到当前归一化状态，得到下一步预测
                    pred_x_norm = current_data.x + pred_delta_x
                    
                    dummy_target_p = torch.zeros(current_data.num_graphs, 1).to(device)
                    
                    # 计算空间权重 (Spatial Weighting)
                    spatial_weights = compute_spatial_weights(current_data.x[:, 0], current_data.batch, axis_weight=10.0, pedestal_weight=5.0)
                    
                    boundary_attr = current_data.boundary if hasattr(current_data, 'boundary') else None
                    freeboundary_attr = current_data.freeboundary if hasattr(current_data, 'freeboundary') else None
                    loss_s = criterion(
                        pred_x_norm,
                        next_data_target.x,
                        current_data,
                        curr_state=current_data.x,
                        boundary=boundary_attr,
                        freeboundary=freeboundary_attr,
                    )
                    loss_accum += loss_s
                    
                    # --- 实时诊断逻辑：遍历整个测试集，收集每个物理时刻的预测/真值包络 ---
                    if do_diagnostics:
                        # 获取物理时间戳 t_now
                        if hasattr(next_data_target, 'time'):
                            t_val = next_data_target.time
                            t_scalar = t_val[0].item() if t_val.numel() > 1 else t_val.item()
                        else:
                            t_scalar = float(t)
                        
                        # 仅提取第一张图 (batch == 0)
                        mask = (current_data.batch == 0)
                        
                        # 反归一化以获取真实的物理量级
                        pred_x_real = pred_x_norm[mask] * global_std + global_mean
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
        if do_diagnostics and epoch % 10 == 0:
            # 1. 生成 Psi (Poloidal Flux) 的 Min/Max 趋势对比图
            times_raw = np.array(diag_data["Psi (Flux)"]["time"], dtype=float)
            pred_min_raw = np.array(diag_data["Psi (Flux)"]["pred_min"], dtype=float)
            pred_max_raw = np.array(diag_data["Psi (Flux)"]["pred_max"], dtype=float)
            truth_min_raw = np.array(diag_data["Psi (Flux)"]["truth_min"], dtype=float)
            truth_max_raw = np.array(diag_data["Psi (Flux)"]["truth_max"], dtype=float)

            # 针对每一个物理时刻 t，聚合整个测试集上所有样本的全局 min/max，
            # 得到 Psi 的真实/预测包络曲线，覆盖整个模拟时间范围。
            if times_raw.size > 0:
                unique_t, inv_idx = np.unique(times_raw, return_inverse=True)
                agg_pred_min = []
                agg_pred_max = []
                agg_truth_min = []
                agg_truth_max = []
                for k in range(len(unique_t)):
                    mask_k = (inv_idx == k)
                    agg_pred_min.append(pred_min_raw[mask_k].min())
                    agg_pred_max.append(pred_max_raw[mask_k].max())
                    agg_truth_min.append(truth_min_raw[mask_k].min())
                    agg_truth_max.append(truth_max_raw[mask_k].max())

                times = unique_t
                pred_min = np.array(agg_pred_min)
                pred_max = np.array(agg_pred_max)
                truth_min = np.array(agg_truth_min)
                truth_max = np.array(agg_truth_max)
            else:
                times = times_raw
                pred_min = pred_min_raw
                pred_max = pred_max_raw
                truth_min = truth_min_raw
                truth_max = truth_max_raw

            plt.figure(figsize=(8, 5))
            plt.plot(times, pred_min, 'r--', label='Pred Min')
            plt.plot(times, pred_max, 'r-', label='Pred Max')
            plt.plot(times, truth_min, 'b--', label='Truth Min')
            plt.plot(times, truth_max, 'b-', label='Truth Max')
            plt.title(f"Psi Min/Max Trend (Epoch {epoch})")
            plt.xlabel("Simulation Time (t_now)")
            plt.ylabel("Psi Value")
            plt.legend()
            plt.grid(True)
            # 将 x 轴范围固定为整个模拟时段，避免只看到局部窗口的时间段
            if GLOBAL_T_MIN < GLOBAL_T_MAX:
                plt.xlim(GLOBAL_T_MIN, GLOBAL_T_MAX)
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
                
            # 这里仍然在标题中保留绝对物理时间，便于与 JOREK 原始输出对照
            if times.size > 0:
                t_last_abs = times[-1]
            else:
                t_last_abs = 0.0
            ax1.set_title(f"True Psi at t={t_last_abs:.5f}")
            fig.colorbar(tc1, ax=ax1)
            ax1.set_aspect('equal', adjustable='box')
            ax2.set_title(f"Predicted Psi at t={times[-1]:.5f}")
            fig.colorbar(tc2, ax=ax2)
            ax2.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, f"field_comparison_epoch_{epoch}.png"))
            plt.close()

            # 3. 基于归一化磁通 psi_n 的 Core / SOL 掩膜可视化
            # 使用当前批次最后一次诊断调用中的 current_data 作为代表快照
            try:
                # 反归一化当前步的 psi 通道（未使用 log 变换）
                psi_norm_full = current_data.x[:, 0] * global_std[0] + global_mean[0]

                # 计算所有节点的归一化磁通 psi_n
                psi_n_all = compute_normalized_flux(
                    psi_norm_full,
                    current_data.batch,
                    getattr(current_data, 'psi_axis', None),
                    getattr(current_data, 'psi_bnd', None),
                    
                )

                # 仅取 batch 中第一个图的节点，用于可视化
                mask_g0 = (current_data.batch == 0)
                pos_g0 = current_data.pos[mask_g0].cpu()
                psi_n_g0 = psi_n_all[mask_g0].detach().cpu().numpy()

                # Core: psi_n <= 1.0, SOL: psi_n > 1.0
                core_mask = psi_n_g0 <= 1.0

                fig_mask, axm = plt.subplots(1, 1, figsize=(6, 5))
                sc = axm.scatter(
                    pos_g0[:, 0].numpy(),
                    pos_g0[:, 1].numpy(),
                    c=psi_n_g0,
                    cmap='viridis',
                    s=3,
                )
                axm.set_title(f"psi_n Core/SOL Mask (Epoch {epoch})")
                axm.set_xlabel("R")
                axm.set_ylabel("Z")
                axm.set_aspect('equal', adjustable='box')
                cbar = fig_mask.colorbar(sc, ax=axm)
                cbar.set_label("psi_n")

                # 用轮廓线粗略标出 psi_n = 1.0 的分界
                try:
                    import matplotlib.tri as mtri

                    tri = None
                    if hasattr(current_data, 'triangles') and current_data.triangles is not None:
                        tri = current_data.triangles.cpu().numpy()

                    if tri is not None:
                        triang = mtri.Triangulation(
                            pos_g0[:, 0].numpy(), pos_g0[:, 1].numpy(), tri
                        )
                        cs = axm.tricontour(
                            triang,
                            psi_n_g0,
                            levels=[1.0],
                            colors='red',
                            linewidths=1.0,
                        )
                        cs.collections[0].set_label("psi_n = 1.0")
                        axm.legend(loc='upper right')
                except Exception:
                    # 若三角剖分失败，则仅保留散点图
                    pass

                plt.tight_layout()
                plt.savefig(os.path.join(VIS_DIR, f"core_sol_mask_epoch_{epoch}.png"))
                plt.close(fig_mask)
            except Exception as e:
                print(f"Warning: failed to generate core/SOL mask visualization: {e}")

    # ================= 5. 最终统计结果可视化 =================
    print("\n[5/5] Generating Final Statistical Visualization...")
    
    # 绘制 5 个变量的最终趋势图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, var_name in enumerate(var_names):
        ax = axes[idx]
        d = diag_data[var_name]
        times_abs = np.array(d["time"], dtype=float)
        pred_min_arr = np.array(d["pred_min"], dtype=float)
        pred_max_arr = np.array(d["pred_max"], dtype=float)
        truth_min_arr = np.array(d["truth_min"], dtype=float)
        truth_max_arr = np.array(d["truth_max"], dtype=float)
        
        # Sort by absolute time
        if times_abs.size > 0:
            sort_idx = np.argsort(times_abs)
            times_sorted = times_abs[sort_idx]
            pred_min_sorted = pred_min_arr[sort_idx]
            pred_max_sorted = pred_max_arr[sort_idx]
            truth_min_sorted = truth_min_arr[sort_idx]
            truth_max_sorted = truth_max_arr[sort_idx]
            
            # Convert to relative time (from first timestep)
            times_rel = times_sorted - times_sorted[0]
        else:
            times_rel = times_abs
            pred_min_sorted = pred_min_arr
            pred_max_sorted = pred_max_arr
            truth_min_sorted = truth_min_arr
            truth_max_sorted = truth_max_arr
        
        # Plot prediction envelope (red) vs ground truth envelope (blue)
        ax.plot(times_rel, pred_min_sorted, 'r--', linewidth=1.5, label='Pred Min')
        ax.plot(times_rel, pred_max_sorted, 'r-', linewidth=2.0, label='Pred Max')
        ax.fill_between(times_rel, pred_min_sorted, pred_max_sorted, alpha=0.1, color='red')
        
        ax.plot(times_rel, truth_min_sorted, 'b--', linewidth=1.5, label='Truth Min', alpha=0.7)
        ax.plot(times_rel, truth_max_sorted, 'b-', linewidth=2.0, label='Truth Max', alpha=0.7)
        ax.fill_between(times_rel, truth_min_sorted, truth_max_sorted, alpha=0.1, color='blue')
        
        ax.set_title(f"{var_name} Evolution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Relative Time (Δt from sequence start)", fontsize=10)
        ax.set_ylabel(f"{var_name} Value", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='best', fontsize=9)

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
    import argparse
    parser = argparse.ArgumentParser(description="Train HyPhy-GLNN Interpolation Test")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_interpolation_test(args.config)
