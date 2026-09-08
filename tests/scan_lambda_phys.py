"""Short λ_phys sweep: train a few hundred steps per value and log val loss."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


LAMBDAS = ["0", "1e-4", "1e-3", "1e-2", "1e-1", "1.0"]


def main():
    results = {}
    for value in LAMBDAS:
        out = Path(f"output/model502_baseline/lambda_scan/lp_{value}")
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, "tests/train_baseline_long.py",
                "--steps", "400",
                "--lambda-phys", value,
                "--output", str(out),
                "--cache", "/tmp/ehl2_cache",
                "--device", "cuda",
            ],
            check=True,
        )
        val_log = json.loads((out / "val_loss.json").read_text())
        results[value] = val_log[-1][1]
    report = Path("output/model502_baseline/lambda_scan/report.json")
    report.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

