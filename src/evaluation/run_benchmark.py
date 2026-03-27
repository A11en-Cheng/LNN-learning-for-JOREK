import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# 保证可以从项目根导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import JorekSequenceDataset, collate_temporal_graphs
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder
from src.evaluation.benchmark import JorekEvaluator


def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())


class HyPhyWrapper:
    """简单包装器：提供 encoder / processor / decoder / stats / dt 属性.

    JorekEvaluator 只需要这些属性即可完成 rollout.
    """

    def __init__(self, encoder, processor, decoder, stats=None, dt: float = 1.0):
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.stats = stats
        self.dt = dt


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run HyPhy-GLNN JOREK benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/interpolation_test_BEST/checkpoints/latest_model.pt",
        help="Path to trained checkpoint (must contain encoder/processor/decoder and global_mean/std)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data dir for benchmark (default: config.data.data_dir)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=40,
        help="Sequence length for each sample (sliding window)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=6,
        help="LNN warmup steps using teacher forcing",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=40,
        help="Autoregressive rollout steps after warmup",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="output/benchmark_BEST",
        help="Directory to save benchmark plots",
    )
    parser.add_argument(
        "--jorek_cpu_hours",
        type=float,
        default=100.0,
        help="Reference JOREK CPU hours per sequence for speedup estimate",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    data_dir = args.data_dir or config["data"]["data_dir"]
    vis_dir = args.vis_dir
    seq_len = args.seq_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Using device: {device}")

    # ================= 1. 数据集 & DataLoader =================
    print(f"[Benchmark] Loading JOREK sequence dataset from {data_dir} (seq_len={seq_len})")
    dataset = JorekSequenceDataset(root_dir=data_dir, seq_len=seq_len)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; please check data_dir and seq_len.")

    # 基准：每个 batch 是一条长序列，方便逐序列统计
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_temporal_graphs,
        num_workers=min(config["training"].get("num_workers", 0), os.cpu_count() or 1),
    )

    # ================= 2. 构建模型并加载权重 =================
    input_dim = config["model"]["input_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    latent_dim = config["model"]["latent_dim"]
    enc_layers = config["model"]["encoder_layers"]
    proc_layers = config["model"]["processor_layers"]
    dec_layers = config["model"]["decoder_layers"]

    print("[Benchmark] Building encoder / processor / decoder...")
    encoder = JorekGNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=enc_layers,
    ).to(device)
    processor = JorekLNNProcessor(
        latent_dim=latent_dim,
        hidden_size=hidden_dim,
        num_layers=proc_layers,
    ).to(device)
    decoder = JorekGNNDecoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=dec_layers,
    ).to(device)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Benchmark] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    def _load_maybe_composite(ckpt_obj, key):
        if isinstance(ckpt_obj, dict) and key in ckpt_obj:
            return ckpt_obj[key]
        return ckpt_obj

    encoder.load_state_dict(_load_maybe_composite(ckpt, "encoder"))
    processor.load_state_dict(_load_maybe_composite(ckpt, "processor"))
    decoder.load_state_dict(_load_maybe_composite(ckpt, "decoder"))

    # 全局归一化统计：优先从 checkpoint 读取
    if "global_mean" not in ckpt or "global_std" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain 'global_mean'/'global_std'. "
            "Please use a physics-informed interpolation checkpoint (train_interpolation_test)."
        )

    global_mean = ckpt["global_mean"].to(device)
    global_std = ckpt["global_std"].to(device)
    stats = {"mean": global_mean, "std": global_std}

    dt = float(config["physics"].get("dt", 1.0))

    model_wrapper = HyPhyWrapper(encoder, processor, decoder, stats=stats, dt=dt)

    # ================= 3. 运行 Benchmark =================
    eval_cfg = {
        "stats": stats,
        "dt": dt,
        "seq_len": seq_len,
        "warmup_steps": args.warmup_steps,
        "rollout_steps": args.rollout_steps,
        "vis_dir": vis_dir,
        "jorek_cpu_hours_per_sequence": args.jorek_cpu_hours,
    }

    evaluator = JorekEvaluator(model_wrapper, eval_cfg, device)

    print("[Benchmark] Running autoregressive rollout and collecting metrics...")
    metrics = evaluator.run_autoregressive_rollout(loader)
    print("[Benchmark] Done. Total sequences:", metrics.get("total_sequences", 0))

    print(f"[Benchmark] Generating plots into {vis_dir}...")
    evaluator.plot_benchmarks()

    print("[Benchmark] Finished. Key metrics summary:")
    print("  Avg wall-clock per sequence (s):", metrics.get("avg_wallclock_per_sequence"))
    print("  Speed-up vs JOREK CPU (x):", metrics.get("speedup_factor_vs_jorek_cpu"))


if __name__ == "__main__":
    main()
