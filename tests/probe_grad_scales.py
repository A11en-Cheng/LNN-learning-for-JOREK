"""Gradient-scale probe: is the physics loss mis-scaled, or does it conflict?

``L = L_data + lambda * L_phys`` failed at every value-calibrated lambda
(pitfalls #36): at lambda = 1e-5 the physics term is only ~5% of the data term by
**value**, yet the run still collapses.  Two possible causes, with completely
different fixes:

* **mis-scaled** - ``||dL_phys/dtheta|| >> ||dL_data/dtheta||`` while the two
  gradients point the same way (cos > 0).  Fix: normalise the weight by the
  measured gradient norms (adaptive lambda).
* **conflicting** - the gradients point in opposite directions (cos < 0).  Scaling
  cannot help; the constraint must be reformulated (weak form, restricted
  channels, or orthogonalised a la PCGrad).

This script measures, per batch, each loss term separately (one term active per
backward pass) and reports the norms, the pairwise cosine, the per-module
breakdown (encoder / CfC cell / decoder), and both the value-based and the
gradient-based lambda that would put the physics term at 10% of the data term.

Example:

    ./.torch-cuda-env/bin/python tests/probe_grad_scales.py \
        --checkpoint output/cfc_interp/gate6_deriv2/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    compute_transform_stats,
)
from src.models.cfc_interp import CfCInterpolator
from src.models.cfc_interp import model_kwargs_from_meta
from tests.train_cfc_interpolation import _batch_loss, _load_channel_scales


def _flat_grad(parameters, loss):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
            for g, p in zip(grads, parameters)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", default="output/cfc_interp/gate6_deriv2/checkpoint.pt")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dt-scale", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    stats = compute_transform_stats(
        {
            name: RunFrameStore(
                Path(manifest["data_root"]) / name, cache_dir=args.cache_dir
            ).load()
            for name in fold["train_runs"]
        },
        runs=fold["train_runs"],
        with_lap=True,
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], fold=fold["fold"], inner="train",
        stats=stats, cache_dir=args.cache_dir, with_shards=False, with_lap=True,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # architecture comes from the checkpoint, not from CLI defaults (they used
    # to disagree: every pre-rewrite checkpoint is an ncps model)
    model = CfCInterpolator(**model_kwargs_from_meta(payload.get("meta") or {}))
    model.load_state_dict(payload["state_dict"])
    model.to(device).train()

    field_mean = torch.tensor(stats["field_mean"], device=device)
    field_std = torch.tensor(stats["field_std"], device=device)
    source_stats = (
        torch.tensor(stats["source_mean"], device=device, dtype=torch.float64),
        torch.tensor(stats["source_std"], device=device, dtype=torch.float64).clamp_min(1e-12),
    )
    increment_var = torch.tensor(stats["increment_var"], device=device, dtype=torch.float32)
    loss_weights = 1.0 / increment_var.clamp_min(1e-30)
    loss_weights = loss_weights / loss_weights.mean()
    channel_scales = _load_channel_scales("auto") or None

    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = {
        "encoder": [p for n, p in model.named_parameters() if n.startswith("encoder")],
        "cell": [
            p for n, p in model.named_parameters()
            if not n.startswith(("encoder", "decoder"))
        ],
        "decoder": [p for n, p in model.named_parameters() if n.startswith("decoder")],
    }
    loaders = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=list,
    )

    def term_grad(items, *, phys: float, h1: float, axis: float):
        items = list(items)
        loss, count, *_ = _batch_loss(
            model, items, device, alpha=None, lambda_h1=h1, lambda_axis=axis,
            lambda_phys=phys, phys_windows=len(items),  # all windows -> deterministic
            stats=(field_mean, field_std), source_stats=source_stats,
            phys_channel_scales=channel_scales, phys_channel_weights=None,
            phys_channel_values=None, phys_detach_denominator=True,
            loss_weights=loss_weights,
        )
        assert count > 0
        grads = {name: _flat_grad(params, loss) for name, params in groups.items()}
        grads["all"] = _flat_grad(parameters, loss)
        return float(loss.detach()), grads

    records: dict = {}
    for batch_index, items in enumerate(loaders):
        if batch_index >= args.batches:
            break
        entries = {
            "data": term_grad(items, phys=0.0, h1=0.0, axis=0.0),
            "h1": term_grad(items, phys=0.0, h1=1.0, axis=0.0),
            "axis": term_grad(items, phys=0.0, h1=0.0, axis=1.0),
            "phys": term_grad(items, phys=1.0, h1=0.0, axis=0.0),
        }
        records.setdefault("loss", []).append({k: v[0] for k, v in entries.items()})
        for term, (_, grads) in entries.items():
            for group, g in grads.items():
                records.setdefault(f"norm_{term}_{group}", []).append(float(g.norm()))
                if group == "all":
                    records.setdefault(f"vec_{term}", []).append(g)

    def med(key):
        return float(np.median(records[key])) if key in records else float("nan")

    def med_loss(term):
        return float(np.median([r[term] for r in records["loss"]]))

    print(f"checkpoint={args.checkpoint} batches={len(records['loss'])}")
    print(f"{'term':8s} {'loss(med)':>12s} {'|grad|':>12s} {'|enc|':>11s} "
          f"{'|cell|':>11s} {'|dec|':>11s}")
    for term in ("data", "h1", "axis", "phys"):
        print(f"{term:8s} {med_loss(term):12.4e} {med(f'norm_{term}_all'):12.4e} "
              f"{med(f'norm_{term}_encoder'):11.3e} {med(f'norm_{term}_cell'):11.3e} "
              f"{med(f'norm_{term}_decoder'):11.3e}")
    print()
    print(f"{'pair':22s} {'cos':>8s} {'|g_b|/|g_a|':>13s}")
    pairs = {}
    for a, b in (("data", "phys"), ("data", "h1"), ("data", "axis"), ("phys", "h1")):
        va = torch.stack(records[f"vec_{a}"]).mean(dim=0)
        vb = torch.stack(records[f"vec_{b}"]).mean(dim=0)
        cos = float((va @ vb) / (va.norm() * vb.norm()).clamp_min(1e-30))
        ratio = float(vb.norm() / va.norm().clamp_min(1e-30))
        pairs[f"{a}_vs_{b}"] = {"cos": cos, "ratio": ratio}
        print(f"{a + ' vs ' + b:22s} {cos:8.4f} {ratio:13.3f}")
    print()
    value_lambda = 0.1 * med_loss("data") / max(med_loss("phys"), 1e-30)
    grad_lambda = 0.1 * med("norm_data_all") / max(med("norm_phys_all"), 1e-30)
    print(f"lambda so the physics term is 10% of the data term BY VALUE    : {value_lambda:.3e}")
    print(f"lambda so the physics GRADIENT is 10% of the data gradient     : {grad_lambda:.3e}")
    print(f"(ratio between the two prescriptions: {grad_lambda / value_lambda:.1f}x)")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "loss": records["loss"],
                    "norms": {k: v for k, v in records.items() if k.startswith("norm_")},
                    "pairs": pairs,
                    "value_lambda_10pct": value_lambda,
                    "gradient_lambda_10pct": grad_lambda,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
