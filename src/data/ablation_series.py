"""Aggregate SPI ablation time series from JOREK HDF5 files.

Empirical contract (verified on the EHL-2 runs): the exported
``xtime_spi_ablation`` / ``xtime_spi_ablation_rate`` datasets are
``(n_steps, 1)`` and hold the **sum over all shards**, not a per-shard series.
The per-shard ablation is available only once per saved frame through
``spi_abl_arr``.  A per-shard, per-internal-step export would require a JOREK
side change; the aggregate series still anchors the total ablated mass.

The helpers here reduce a dataset to one value per internal step, query the
cumulative value at a frame time, and return the integrated increment between
two frame times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class AblationSeries:
    times: np.ndarray  # internal-step times in JOREK units
    cumulative: np.ndarray  # total ablated atoms per step (monotone)
    rate_total: np.ndarray  # total ablation rate [atom/s] per step
    path: Path

    def index_at(self, time: float) -> int:
        """Last internal step at or before ``time`` (clamped to the series)."""
        index = int(np.searchsorted(self.times, time, side="right")) - 1
        return int(np.clip(index, 0, len(self.times) - 1))

    def cumulative_at(self, time: float) -> float:
        return float(self.cumulative[self.index_at(time)])

    def increment(self, t_start: float, t_end: float) -> float:
        """Total ablated atoms deposited in ``(t_start, t_end]``."""
        return self.cumulative_at(t_end) - self.cumulative_at(t_start)


def _reduce_to_steps(dataset, num_steps: int, name: str) -> np.ndarray:
    """Reduce a 1-D or 2-D dataset to one total value per internal step."""
    values = np.asarray(dataset, dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] != num_steps:
            raise ValueError(f"{name}: length {values.shape[0]} != n_steps {num_steps}")
        return values
    if values.ndim == 2:
        if values.shape[0] == num_steps:
            return values.sum(axis=1)
        if values.shape[1] == num_steps:
            return values.sum(axis=0)
        raise ValueError(f"{name}: shape {values.shape} is not compatible with n_steps {num_steps}")
    raise ValueError(f"{name}: expected 1-D or 2-D data, got shape {values.shape}")


def read_ablation_series(path: str | Path) -> AblationSeries | None:
    """Read the aggregate ablation series from a ``jorek*.h5`` file.

    Returns ``None`` when the file has no SPI ablation bookkeeping (e.g. a
    frame written before injection).
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "xtime" not in handle or "xtime_spi_ablation" not in handle:
            return None
        times_full = np.asarray(handle["xtime"][()], dtype=np.float64).reshape(-1)
        num_steps_full = times_full.shape[0]
        index_now = (
            int(np.ravel(handle["index_now"][()])[0])
            if "index_now" in handle
            else num_steps_full
        )
        num_steps = min(num_steps_full, index_now)
        times = times_full[:num_steps]
        cumulative = _reduce_to_steps(
            handle["xtime_spi_ablation"][()], num_steps_full, "xtime_spi_ablation"
        )[:num_steps]
        if "xtime_spi_ablation_rate" in handle:
            rate_raw = np.asarray(handle["xtime_spi_ablation_rate"][()])
            rate_total = _reduce_to_steps(
                rate_raw, num_steps_full, "xtime_spi_ablation_rate"
            )[:num_steps]
        else:
            rate_total = np.zeros(num_steps, dtype=np.float64)
    if num_steps == 0:
        return None
    return AblationSeries(times=times, cumulative=cumulative, rate_total=rate_total, path=path)


def frame_totals(path: str | Path) -> tuple[float, float, float] | None:
    """Return ``(t_last, cumulative_total, rate_total)`` of one frame file."""
    series = read_ablation_series(path)
    if series is None or len(series.times) == 0:
        return None
    return (
        float(series.times[-1]),
        float(series.cumulative[-1]),
        float(series.rate_total[-1]),
    )


__all__ = ["AblationSeries", "frame_totals", "read_ablation_series"]
