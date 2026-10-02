"""Does one CfC step over dt equal a chain of finer sub-steps?

Why this matters
----------------
The project's target is *non-uniform time interpolation / prediction*: the
network must be usable at any ``t*`` between two recorded frames, and the
physics loss is supposed to correct that interpolation.  That only makes sense
if the integrator is (at least approximately) time-consistent, i.e. if

    advance(dt)  ~=  advance(a * dt) o advance((1 - a) * dt)

for every split ``a``.  The ncps CfC cell used here is *not* an ODE propagator
in that strict sense: in ``mode="default"`` the update is

    h_new = ff1 * (1 - t) + t * ff2,      t = sigmoid(t_a * ts + t_b)

with ``ff1, ff2, t_a, t_b`` functions of ``(x, h_old)``.  As ``ts -> 0`` the
update does **not** tend to the identity, so splitting an interval is not a
discretisation refinement: it is *more recurrent updates*.  This script
measures how large that split ambiguity is compared with the increment the
network is asked to predict, on a trained checkpoint.

Outputs (standardised model space, per field and pooled; ``err`` is normalised by
``||u_{k+1} - u_k||`` of the same field, so ``dR2 = 1 - err^2``):

  * ``err vs truth``  — how wrong each integrator is on the recorded increment
    (``1step`` is a single cell call over ``dt``; ``S=..`` are equal chains; the
    ``S`` the checkpoint was trained with is the number the eval script reports);
  * ``diff vs 1step`` — how much the integrator choice changes the answer, i.e.
    the split ambiguity that decides whether ``--sub-steps`` is a *model* choice
    (baked into training) or a *discretisation* refinement.

Example:

    ./.torch-cuda-env/bin/python tests/probe_time_consistency.py \
        --checkpoint output/cfc_interp/baseline_dataonly/checkpoint.pt --max-windows 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.data.model502_schema import FIELD_NAMES
from src.models.cfc_interp import CfCInterpolator
from src.models.cfc_interp import deterministic_fractions
from src.models.cfc_interp import model_kwargs_from_meta


def _per_field_sq(tensor: torch.Tensor) -> torch.Tensor:
    """Per-field squared Frobenius norm of a ``[nodes, fields]`` tensor."""
    return tensor.double().pow(2).sum(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", default="output/cfc_interp/baseline_dataonly/checkpoint.pt")
    # architecture flags are informational only: they come from the checkpoint
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--burn-in-substeps", type=int, default=1)
    parser.add_argument("--alphas", default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--chains", default="2,4,8,25")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    stores = {
        name: RunFrameStore(Path(manifest["data_root"]) / name, cache_dir=args.cache_dir).load()
        for name in fold["train_runs"]
    }
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # architecture comes from the checkpoint (legacy ncps vs the LTC rewrite)
    model_kwargs = model_kwargs_from_meta(payload.get("meta") or {})
    with_lap = bool(model_kwargs["input_deriv2"])
    stats = compute_transform_stats(stores, runs=fold["train_runs"], with_lap=with_lap)
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out], inner="val",
        stats=stats, cache_dir=args.cache_dir, with_shards=False, with_lap=with_lap,
    )
    model = CfCInterpolator(**model_kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    alphas = [float(value) for value in args.alphas.split(",") if value]
    chains = [int(value) for value in args.chains.split(",") if value]
    n_fields = len(FIELD_NAMES)
    labels = [f"1step"] + [f"a={value}" for value in alphas] + [f"S={value}" for value in chains]
    accumulated = {label: torch.zeros(n_fields, dtype=torch.float64, device=device) for label in labels}
    errors = {label: torch.zeros(n_fields, dtype=torch.float64, device=device) for label in labels}
    increments = torch.zeros(n_fields, dtype=torch.float64, device=device)
    pooled_accumulated = {label: 0.0 for label in labels}
    pooled_errors = {label: 0.0 for label in labels}
    pooled_increments = 0.0
    worst: list[dict] = []
    windows = 0
    with torch.no_grad():
        for index in range(min(len(dataset), args.max_windows)):
            item = dataset[index]
            batch = to_device(collate_windows([item]), device)
            u, dt, source = batch["u"][0], batch["dt"][0], batch["source"][0]
            deriv = batch["u_deriv"][0]
            lap_feat = batch["u_lap"][0]
            burn_in = int((~batch["loss_mask"][0]).sum().item())
            hidden = None
            for k in range(burn_in):
                _, _, hidden = model.step(
                    u[k], hidden, dt[k], source[k + 1], None, deriv[k], lap_feat[k],
                    sub_steps=args.burn_in_substeps,
                )
            k0 = burn_in
            base = u[k0]
            increment_sq = _per_field_sq(u[k0 + 1] - base)
            increments = increments + increment_sq
            pooled_increments += float(increment_sq.sum())
            reference = model.multi_substep(
                base, hidden, deterministic_fractions(1), dt[k0], source[k0 + 1], None,
                deriv[k0], lap_feat[k0],
            )["prediction"]
            differences = {"1step": _per_field_sq(reference - u[k0 + 1])}
            truth_errors = {"1step": differences["1step"]}
            for alpha in alphas:
                split = model.multi_substep(
                    base, hidden, deterministic_fractions(2, alpha), dt[k0], source[k0 + 1],
                    None, deriv[k0], lap_feat[k0],
                )["prediction"]
                differences[f"a={alpha}"] = _per_field_sq(split - reference)
                truth_errors[f"a={alpha}"] = _per_field_sq(split - u[k0 + 1])
            for sub_steps in chains:
                chain = model.multi_substep(
                    base, hidden, deterministic_fractions(sub_steps), dt[k0], source[k0 + 1],
                    None, deriv[k0], lap_feat[k0],
                )["prediction"]
                differences[f"S={sub_steps}"] = _per_field_sq(chain - reference)
                truth_errors[f"S={sub_steps}"] = _per_field_sq(chain - u[k0 + 1])
            for label in labels:
                accumulated[label] = accumulated[label] + differences[label]
                pooled_accumulated[label] += float(differences[label].sum())
                errors[label] = errors[label] + truth_errors[label]
                pooled_errors[label] += float(truth_errors[label].sum())
            worst.append({
                "run": item["run"], "index": index,
                "one_step_sq": [float(v) for v in differences["1step"]],
            })
            windows += 1

    safe = increments.clamp_min(1e-30)
    pooled_scale = max(pooled_increments, 1e-30)

    def table(title: str, store: dict, pooled: dict) -> None:
        print(f"\n{title}")
        print(f"{'field':10s}" + "".join(f"{label:>9s}" for label in labels))
        for index, name in enumerate(FIELD_NAMES):
            print(
                f"{name:10s}"
                + "".join(
                    f"{float((store[label][index] / safe[index]).sqrt()):9.3f}" for label in labels
                )
            )
        print(
            f"{'POOLED':10s}"
            + "".join(f"{float((pooled[label] / pooled_scale) ** 0.5):9.3f}" for label in labels)
        )

    print(f"held-out={args.held_out} windows={windows} checkpoint={args.checkpoint}")
    table("err vs truth  (= sqrt(1 - dR2), 1.0 = persistence)", errors, pooled_errors)
    table("diff vs 1step (integrator ambiguity)", accumulated, pooled_accumulated)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "windows": windows,
                    "labels": labels,
                    "diff_vs_1step": {label: accumulated[label].tolist() for label in labels},
                    "err_vs_truth": {label: errors[label].tolist() for label in labels},
                    "increment_sq": increments.tolist(),
                    "pooled_diff": {label: pooled_accumulated[label] for label in labels},
                    "pooled_err": {label: pooled_errors[label] for label in labels},
                    "pooled_increment_sq": pooled_increments,
                    "per_window": worst,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
