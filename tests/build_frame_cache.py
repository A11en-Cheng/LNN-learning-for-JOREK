"""Parse every aligned frame once and cache compact tensors to a local folder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser


def build(cache_root="/tmp/ehl2_cache", data_root="data/EHL-2"):
    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    runs = [r["name"] for r in manifest["runs"] if r["split"] in {"train", "val"}]
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for run in runs:
        (root / run).mkdir(parents=True, exist_ok=True)
        for frame in aligned_frames(Path(data_root) / run):
            path = root / run / f"{frame.index}.pt"
            if path.exists():
                total += 1
                continue
            graph = JorekDataParser(str(frame.state_path), require_source=True).parse_to_graph()
            torch.save(
                {
                    "x": graph.x,
                    "edge_index": graph.edge_index,
                    "edge_attr": graph.edge_attr,
                    "pos": graph.pos,
                    "time": graph.time,
                    "t_norm": graph.t_norm,
                    "condition": graph.condition,
                    "source": graph.source if hasattr(graph, "source") else None,
                    "boundary": graph.boundary if hasattr(graph, "boundary") else None,
                    "freeboundary": graph.freeboundary if hasattr(graph, "freeboundary") else None,
                },
                path,
            )
            total += 1
    print("cached frames:", total, "->", root)


if __name__ == "__main__":
    build()
