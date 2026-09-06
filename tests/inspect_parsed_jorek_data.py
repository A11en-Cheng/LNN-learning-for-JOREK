import os
import sys
from typing import Any

import torch

# 确保可以导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser


def _format_scalar_tensor(t: torch.Tensor) -> str:
    """将单元素 Tensor 转为字符串。"""
    try:
        return f"{t.item()} (dtype={t.dtype}, shape={tuple(t.shape)})"
    except Exception:
        return f"Tensor(dtype={t.dtype}, shape={tuple(t.shape)})"


def summarize_data_object(data: Any) -> None:
    """按属性类型打印 Data 对象内容。

    - 网格 / 空间张量: 只打印维度 (shape)。
    - 标量 / 布尔 / 列表: 打印原始值。
    - 特别突出 time 属性。
    """
    from torch_geometric.data import Data

    if not isinstance(data, Data):
        print(f"[WARN] Expected torch_geometric.data.Data, got {type(data)}")
        return

    print("  Basic graph stats:")
    print(f"    num_nodes : {data.num_nodes}")
    print(f"    num_edges : {data.num_edges}")

    # 先单独处理时间属性
    if hasattr(data, "time"):
        t = getattr(data, "time")
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                print("  time attribute:")
                print(f"    time = {_format_scalar_tensor(t)}")
            else:
                print("  time attribute (tensor with multiple elements):")
                print(f"    time shape = {tuple(t.shape)}, dtype={t.dtype}")
        else:
            print("  time attribute (non-tensor):")
            print(f"    time = {t!r}")
    else:
        print("  [INFO] data.time not present")

    print("\n  Node / edge tensors (only shapes):")
    for key in ["x", "pos", "edge_index", "edge_attr", "triangles", "boundary"]:
        if hasattr(data, key):
            val = getattr(data, key)
            if isinstance(val, torch.Tensor):
                print(f"    {key:12s}: shape={tuple(val.shape)}, dtype={val.dtype}")

    print("\n  Other attributes:")
    # 遍历 data 中的所有键，过滤掉上面已经打印过的
    # 注意：在当前 PyG 版本中，Data.keys 是一个方法，需要调用 data.keys()
    printed_keys = {"x", "pos", "edge_index", "edge_attr", "triangles", "boundary", "time"}
    for key in data.keys():
        if key in printed_keys:
            continue
        val = getattr(data, key)

        # 标量 Tensor
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            print(f"    {key:12s}: {_format_scalar_tensor(val)}")
        # 多元素 Tensor，当作网格型数据，只打印维度
        elif isinstance(val, torch.Tensor):
            print(f"    {key:12s}: Tensor(shape={tuple(val.shape)}, dtype={val.dtype})")
        # Python 标量/布尔
        elif isinstance(val, (int, float, bool, str)):
            print(f"    {key:12s}: {val!r}")
        # 列表或元组，直接打印原始值
        elif isinstance(val, (list, tuple)):
            print(f"    {key:12s}: list(len={len(val)}): {val!r}")
        # 字典，打印键和值
        elif isinstance(val, dict):
            print(f"    {key:12s}: dict with keys={list(val.keys())!r}")
            for k2, v2 in val.items():
                print(f"        - {k2}: {v2!r}")
        else:
            print(f"    {key:12s}: {type(val)} -> {val!r}")


def inspect_jorek_directory(root_dir: str, max_files: int | None = 5) -> None:
    """遍历 root_dir 下的 jorek*.h5 文件，逐个解析并打印 Data 信息。"""
    import glob
    import re

    pattern = os.path.join(root_dir, "jorek*.h5")
    files = glob.glob(pattern)

    if not files:
        print(f"[ERROR] No HDF5 files found under: {root_dir}")
        return

    # 按文件名中的数字排序，与数据集逻辑保持一致
    def extract_number(path: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    files = sorted(files, key=extract_number)

    if max_files is not None:
        files = files[:max_files]

    print(f"Found {len(files)} HDF5 files to inspect in {root_dir}\n")

    for idx, fp in enumerate(files):
        print("=" * 80)
        print(f"[{idx:03d}] File: {os.path.basename(fp)}")
        print("=" * 80)
        try:
            parser = JorekDataParser(fp)
            data = parser.parse_to_graph()
        except Exception as e:
            print(f"  [ERROR] Failed to parse file: {e}")
            continue

        summarize_data_object(data)
        print("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect JOREK HDF5 -> PyG Data contents")
    parser.add_argument(
        "--root",
        type=str,
        default="data/EHL-2/scan_1d22_ang12_massloss_neon",
        help="Directory containing jorek*.h5 files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of files to inspect (None for all)",
    )

    args = parser.parse_args()
    max_files = args.max_files if args.max_files >= 0 else None

    print(f"Inspecting JOREK data under: {args.root}")
    inspect_jorek_directory(args.root, max_files=max_files)
