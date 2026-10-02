"""Torch dataset, frame store and collate for the Phase-1 window corpus.

The dataset consumes the manifest produced by :mod:`src.data.window_manifest`
and yields, for every window, the tensors the interpolation trainer needs:

* ``u_raw`` / ``u``: ``[T, nodes, 9]`` canonical state (raw and normalised),
  where the first ``burn_in`` frames are inputs only;
* ``condition``: ``[T, D]`` zero-dimensional parameter channel per frame;
* ``source`` / ``source_available``: ``[T, nodes, 5]`` marker projections
  (teacher source) with an explicit availability flag;
* ``shard_values`` / ``shard_mask``: padded shard-level ``p(t)`` set per frame;
* ``t`` / ``dt``: JOREK-unit frame times and interval lengths;
* ``topology``: the shared poloidal graph (``pos``/``edge_index``/``edge_attr``).

Per-run frames are loaded once into memory (optionally cached on disk).
Statistics passed to ``stats`` must be computed on the training windows only;
:func:`compute_transform_stats` does that for a set of runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.data_manifest import aligned_frames
from src.data.ablation_series import frame_totals
from src.data.fragment_state import SHARD_FIELD_NAMES, read_shard_state
from src.data.jorek_h5_parser import JorekDataParser
from src.data.nonlocal_table import nonlocal_table_from_frame
from src.data.model502_schema import (
    FIELD_NAMES,
    LOG1P_FIELD_INDICES,
    MARKER_PROJECTION_FIELD_NAMES,
    reorder_h5_values,
    standardize,
    transform_state,
)
from src.data.parameter_channel import CONDITION_DIM, build_condition_vector
from src.physics.spi_model import NonlocalAblationTable

NUM_STATE_FIELDS = len(FIELD_NAMES)
NUM_SOURCE_FIELDS = len(MARKER_PROJECTION_FIELD_NAMES)


@dataclass(frozen=True)
class RunTopology:
    """Shared poloidal graph of one run (identical across its frames)."""

    run: str
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    boundary: torch.Tensor
    size: torch.Tensor = None          # [4, 4, n_elements] Hermite dof scaling
    vertex: torch.Tensor = None        # [4, n_elements] 0-based node ids
    x_dofs: torch.Tensor = None        # [nodes, 4, 2] geometry dofs


def load_run_topology(run_dir: str | Path) -> RunTopology:
    """Build the graph topology once per run by parsing its first aligned frame."""
    run_dir = Path(run_dir)
    frames = aligned_frames(run_dir)
    if not frames:
        raise FileNotFoundError(f"no aligned jorek/part frames in {run_dir}")
    graph = JorekDataParser(str(frames[0].state_path)).parse_to_graph()
    with h5py.File(frames[0].state_path, "r") as handle:
        size = np.asarray(handle["size"][()], dtype=np.float32)
        vertex = np.asarray(handle["vertex"][()], dtype=np.int64) - 1
        x_raw = np.asarray(handle["x"][()], dtype=np.float32)  # [2, 4, 1, nodes]
        x_dofs = np.transpose(x_raw[:, :, 0, :], (2, 1, 0))    # [nodes, 4, 2]
    return RunTopology(
        run=run_dir.name,
        pos=graph.pos.float().contiguous(),
        edge_index=graph.edge_index.long().contiguous(),
        edge_attr=graph.edge_attr.float().contiguous(),
        boundary=graph.boundary.long().contiguous(),
        size=torch.as_tensor(size),
        vertex=torch.as_tensor(vertex),
        x_dofs=torch.as_tensor(x_dofs),
    )


def _namelist_text(run_dir: Path) -> str | None:
    for candidate in ("jorek_namelist_spitest", "jorek_namelist"):
        path = run_dir / candidate
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def _projection_path(state_path: Path) -> Path | None:
    match = state_path.name.startswith("jorek")
    if not match:
        return None
    index = int(state_path.stem.replace("jorek", ""))
    candidate = state_path.parent / f"projections{index:06d}.h5"
    return candidate if candidate.exists() else None


def read_frame_payload(
    state_path: str | Path,
    *,
    namelist_text: str | None = None,
    with_shards: bool = True,
) -> dict:
    """Read the light per-frame payload needed by the window dataset."""
    state_path = Path(state_path)
    with h5py.File(state_path, "r") as handle:
        raw = handle["values"]
        if raw.ndim != 4 or raw.shape[2] != 1:
            raise ValueError(
                f"{state_path.name}: expected 2D values [n_var, 4, 1, nodes], got {raw.shape}"
            )
        values_raw = raw[:, 0, 0, :]
        # The H5 ``values`` dataset stores all four bicubic-Hermite dofs per node
        # (value, d/ds, d/dt, d2/dsdt); keep them so the FEM-consistent
        # derivative module can use JOREK's own derivatives.
        u_dofs = np.transpose(np.asarray(raw[:, :, 0, :]), (2, 1, 0))  # [nodes, 4, var]
        if values_raw.shape[0] < NUM_STATE_FIELDS:
            raise ValueError(
                f"{state_path.name}: expected >= {NUM_STATE_FIELDS} channels, got {values_raw.shape[0]}"
            )
        u = reorder_h5_values(torch.as_tensor(values_raw, dtype=torch.float32))
        t_now = float(np.ravel(handle["t_now"][()])[0]) if "t_now" in handle else float("nan")
        t_norm = float(np.ravel(handle["t_norm"][()])[0]) if "t_norm" in handle else 1.0
        psi_axis = (
            float(np.ravel(handle["psi_axis_t"][()])[-1]) if "psi_axis_t" in handle else float("nan")
        )
        psi_bnd = (
            float(np.ravel(handle["psi_bnd_t"][()])[-1]) if "psi_bnd_t" in handle else float("nan")
        )
        condition = build_condition_vector(handle, namelist_text=namelist_text)

    source = None
    source_available = False
    projection_path = _projection_path(state_path)
    if projection_path is not None:
        with h5py.File(projection_path, "r") as handle:
            projection = handle["values"][:NUM_SOURCE_FIELDS, 0, 0, :]
            if projection.shape[0] != NUM_SOURCE_FIELDS:
                raise ValueError(
                    f"{projection_path.name}: expected {NUM_SOURCE_FIELDS} projection fields, "
                    f"got {projection.shape[0]}"
                )
            source = torch.as_tensor(projection, dtype=torch.float32).transpose(0, 1).contiguous()
            source_available = True

    shard_values = shard_alive = None
    if with_shards:
        shard_state = read_shard_state(state_path)
        if shard_state is not None:
            shard_values = torch.as_tensor(shard_state.values, dtype=torch.float32)
            shard_alive = torch.as_tensor(shard_state.alive, dtype=torch.bool)

    return {
        "u": u,
        "u_dofs": torch.as_tensor(u_dofs, dtype=torch.float32),
        "time": torch.tensor([t_now], dtype=torch.float32),
        "t_norm": torch.tensor([t_norm], dtype=torch.float32),
        "psi_axis": psi_axis,
        "psi_bnd": psi_bnd,
        "condition": torch.as_tensor(condition["vector"], dtype=torch.float32),
        "condition_names": tuple(condition["names"]),
        "source": source,
        "source_available": source_available,
        "shard_values": shard_values,
        "shard_alive": shard_alive,
    }


class RunFrameStore:
    """All aligned frames of one run, loaded once (optionally disk-cached)."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
        with_shards: bool = True,
        with_lap: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run = self.run_dir.name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.with_shards = with_shards
        self.frames = aligned_frames(self.run_dir)
        self.topology: RunTopology | None = None
        self.u: torch.Tensor | None = None
        self.u_dofs: torch.Tensor | None = None
        self.time: torch.Tensor | None = None
        self.t_norm: torch.Tensor | None = None
        self.psi_axis: torch.Tensor | None = None
        self.psi_bnd: torch.Tensor | None = None
        self.condition: torch.Tensor | None = None
        self.source: torch.Tensor | None = None
        self.source_available: torch.Tensor | None = None
        self.shard_values: list[torch.Tensor | None] = []
        self.shard_alive: list[torch.Tensor | None] = []
        self.ablation_cumulative: torch.Tensor | None = None
        self.ablation_rate_total: torch.Tensor | None = None
        self._nonlocal_tables: dict[int, NonlocalAblationTable] = {}
        self._deriv: torch.Tensor | None = None
        self._lap: torch.Tensor | None = None
        self._loaded = False

    @property
    def cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.run}.pt"

    def load(self) -> "RunFrameStore":
        if self._loaded:
            return self
        cache_path = self.cache_path
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if "ablation_cumulative" in payload and "psi_axis" in payload and "u_dofs" in payload:
                condition = payload.get("condition")
                if (
                    condition is not None
                    and condition.shape[-1] == CONDITION_DIM
                ):
                    self._restore(payload)
                    self._loaded = True
                    return self
                # The parameter channel grew (e.g. ZK_par_max / T_min / D_par
                # were added): rebuild instead of serving a stale layout.

        namelist_text = _namelist_text(self.run_dir)
        us: list[torch.Tensor] = []
        u_dofs: list[torch.Tensor] = []
        times: list[torch.Tensor] = []
        t_norms: list[torch.Tensor] = []
        conditions: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        source_available: list[bool] = []
        shard_values: list[torch.Tensor | None] = []
        shard_alive: list[torch.Tensor | None] = []
        ablation_cumulative: list[float] = []
        ablation_rate_total: list[float] = []
        psi_axis: list[float] = []
        psi_bnd: list[float] = []
        for frame in self.frames:
            payload = read_frame_payload(
                frame.state_path, namelist_text=namelist_text, with_shards=self.with_shards
            )
            us.append(payload["u"])
            u_dofs.append(payload["u_dofs"])
            times.append(payload["time"])
            t_norms.append(payload["t_norm"])
            psi_axis.append(float(payload["psi_axis"]))
            psi_bnd.append(float(payload["psi_bnd"]))
            conditions.append(payload["condition"])
            sources.append(
                payload["source"]
                if payload["source"] is not None
                else torch.zeros(payload["u"].shape[0], NUM_SOURCE_FIELDS)
            )
            source_available.append(bool(payload["source_available"]))
            shard_values.append(payload["shard_values"])
            shard_alive.append(payload["shard_alive"])
            totals = frame_totals(frame.state_path)
            if totals is None:
                ablation_cumulative.append(float("nan"))
                ablation_rate_total.append(float("nan"))
            else:
                _, cumulative, rate = totals
                ablation_cumulative.append(cumulative)
                ablation_rate_total.append(rate)

        self.u = torch.stack(us)
        self.u_dofs = torch.stack(u_dofs)
        self.time = torch.cat(times)
        self.t_norm = torch.cat(t_norms)
        self.psi_axis = torch.tensor(psi_axis, dtype=torch.float64)
        self.psi_bnd = torch.tensor(psi_bnd, dtype=torch.float64)
        self.condition = torch.stack(conditions)
        self.source = torch.stack(sources)
        self.source_available = torch.tensor(source_available, dtype=torch.bool)
        self.shard_values = shard_values
        self.shard_alive = shard_alive
        self.ablation_cumulative = torch.tensor(ablation_cumulative, dtype=torch.float64)
        self.ablation_rate_total = torch.tensor(ablation_rate_total, dtype=torch.float64)
        self.topology = load_run_topology(self.run_dir)
        self._loaded = True
        self._save()
        return self

    def _save(self) -> None:
        cache_path = self.cache_path
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "run": self.run,
                "u": self.u,
                "u_dofs": self.u_dofs,
                "time": self.time,
                "t_norm": self.t_norm,
                "psi_axis": self.psi_axis,
                "psi_bnd": self.psi_bnd,
                "condition": self.condition,
                "source": self.source,
                "source_available": self.source_available,
                "shard_values": self.shard_values,
                "shard_alive": self.shard_alive,
                "ablation_cumulative": self.ablation_cumulative,
                "ablation_rate_total": self.ablation_rate_total,
                "topology": self.topology,
            },
            cache_path,
        )

    def _restore(self, payload: dict) -> None:
        self.u = payload["u"]
        self.u_dofs = payload["u_dofs"]
        self.time = payload["time"]
        self.t_norm = payload["t_norm"]
        self.psi_axis = payload["psi_axis"]
        self.psi_bnd = payload["psi_bnd"]
        self.condition = payload["condition"]
        self.source = payload["source"]
        self.source_available = payload["source_available"]
        self.shard_values = list(payload["shard_values"])
        self.shard_alive = list(payload["shard_alive"])
        self.ablation_cumulative = payload["ablation_cumulative"]
        self.ablation_rate_total = payload["ablation_rate_total"]
        self.topology = payload["topology"]

    def __len__(self) -> int:
        return len(self.frames)

    def nonlocal_table(self, position: int) -> NonlocalAblationTable:
        """Rebuild (once) the frame's ``nonlocal_array`` surface table.

        The on-disk file is only the last snapshot, so the table is recomputed
        from this frame's fields; results are cached per position.
        """
        table = self._nonlocal_tables.get(position)
        if table is None:
            table = nonlocal_table_from_frame(self.frames[position].state_path)
            self._nonlocal_tables[position] = table
        return table

    def derivative_features(self) -> torch.Tensor:
        """First-order local-derivative features ``[T, nodes, 2 * n_var]``.

        Computed once per run from JOREK's stored Hermite dofs with the static
        FEM gradient map; cached on the store.
        """
        if self._deriv is None:
            from src.physics.fem_derivatives import build_gradient_map, nodal_gradient_torch

            if self.topology is None:
                self.topology = load_run_topology(self.run_dir)
            if self.u_dofs is None:
                raise RuntimeError("frame store has no dof data")
            topology = self.topology
            gradient_map, vertex = build_gradient_map(
                topology.size.numpy(), topology.vertex.numpy(), topology.x_dofs.numpy()
            )
            gradient_map = torch.as_tensor(gradient_map, dtype=self.u_dofs.dtype)
            vertex = torch.as_tensor(vertex)
            frames = []
            for index in range(self.u_dofs.shape[0]):
                grad = nodal_gradient_torch(
                    self.u_dofs[index], gradient_map, vertex, n_nodes=self.u_dofs.shape[1]
                )
                frames.append(grad.reshape(grad.shape[0], -1))
            self._deriv = torch.stack(frames)
        return self._deriv

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["topology"] = None
        state["_nonlocal_tables"] = {}
        state["_deriv"] = None
        state["_lap"] = None
        return state

    def laplacian_features(self) -> torch.Tensor:
        """Graph-LSQ Laplacian of every field ``[T, nodes, n_var]``.

        Uses the same ``div(grad(...))`` operator the physics residuals use
        (verified against the recorded ``Z_j``/``omega`` channels at
        corr 0.987-0.997, see ``tests/check_fem_second_order.py``).  The operator
        annihilates constants, so standardising the field and then taking the
        Laplacian equals z-scoring the physical Laplacian.

        .. note:: the *dof-based* FEM Laplacian in ``fem_derivatives`` is broken
           by a large factor (median 1.5e3 vs an expected ~2 on psi); do not use
           it until that is fixed.
        """
        if self._lap is None:
            from src.physics.grad_ops import compute_divergence, compute_gradient

            if self.topology is None:
                self.topology = load_run_topology(self.run_dir)
            topology = self.topology
            edges = topology.edge_index
            attrs = topology.edge_attr
            frames = []
            for index in range(self.u.shape[0]):
                values = self.u[index].float()
                # ``compute_divergence`` takes one vector per node, so the
                # Laplacian is assembled field by field (same call pattern as
                # ``model502_full_loss._laplacian``).
                per_field = [
                    compute_divergence(
                        compute_gradient(values[:, field], edges, attrs),
                        edges,
                        attrs,
                    )
                    for field in range(values.shape[1])
                ]
                frames.append(torch.stack(per_field, dim=1))
            self._lap = torch.stack(frames)
        return self._lap


def _pad_shards(
    frames: Sequence[int],
    store: RunFrameStore,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the variable-length shard sets of one window to ``[T, S_max, F]``."""
    counts = []
    for position in frames:
        values = store.shard_values[position] if store.shard_values else None
        counts.append(0 if values is None else int(values.shape[0]))
    max_shards = max(counts, default=0)
    num_frames = len(frames)
    if max_shards == 0:
        return (
            torch.zeros(num_frames, 0, len(SHARD_FIELD_NAMES)),
            torch.zeros(num_frames, 0, dtype=torch.bool),
        )
    values = torch.zeros(num_frames, max_shards, len(SHARD_FIELD_NAMES))
    mask = torch.zeros(num_frames, max_shards, dtype=torch.bool)
    for row, position in enumerate(frames):
        frame_values = store.shard_values[position]
        frame_alive = store.shard_alive[position]
        if frame_values is None:
            continue
        n = int(frame_values.shape[0])
        values[row, :n] = frame_values
        if frame_alive is not None:
            mask[row, :n] = frame_alive
        else:
            mask[row, :n] = True
    return values, mask


class InterpolationWindowDataset(Dataset):
    """Windows of the Phase-1 corpus, with optional train-only normalisation."""

    def __init__(
        self,
        manifest: dict | str | Path,
        data_root: str | Path,
        *,
        inner: str | None = None,
        fold: int | None = None,
        runs: Iterable[str] | None = None,
        normalize: bool = True,
        stats: dict | None = None,
        cache_dir: str | Path | None = None,
        with_shards: bool = True,
        with_lap: bool = False,
        with_condition: bool = False,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.manifest = manifest
        self.data_root = Path(data_root)
        self.normalize = normalize
        self.stats = stats
        self.with_shards = bool(with_shards)
        # Second-order (graph-LSQ Laplacian) features are opt-in: they need the
        # element geometry (``size``/``vertex``/``x``) and cost one div(grad) per
        # field per frame.
        self.with_lap = bool(with_lap)
        # ``with_condition`` standardises the 61-dim zero-dimensional parameter
        # channel with train-split statistics so the model can consume it as an
        # input feature (physical coefficients differ between runs/scenarios).
        self.with_condition = bool(with_condition)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.held_out_run: str | None = None
        allowed_runs = set(runs) if runs is not None else None
        if fold is not None:
            fold_record = next(
                (record for record in manifest["folds"] if record["fold"] == fold), None
            )
            if fold_record is None:
                raise ValueError(f"fold {fold} is not in the manifest")
            self.held_out_run = fold_record["held_out_run"]
            allowed_runs = set(fold_record["train_runs"])

        self.windows: list[dict] = []
        for record in manifest["windows"]:
            if inner is not None and record["inner"] != inner:
                continue
            if allowed_runs is not None and record["run"] not in allowed_runs:
                continue
            self.windows.append(record)
        self._stores: dict[str, RunFrameStore] = {}

    def _store(self, run: str) -> RunFrameStore:
        if run not in self._stores:
            store = RunFrameStore(self.data_root / run, cache_dir=self.cache_dir)
            self._stores[run] = store.load()
        return self._stores[run]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        record = self.windows[index]
        store = self._store(record["run"])
        if store.u is None or store.time is None or store.condition is None:
            raise RuntimeError(f"frame store for {record['run']} is not loaded")
        frames = record["frame_indices"]
        u_raw = store.u[frames]
        condition = store.condition[frames]
        time = store.time[frames]
        dt = time[1:] - time[:-1]
        # Fail fast on non-monotonic windows: a negative dt reaches the LTC
        # timespan, overflows ``exp`` and turns the whole run into NaN.  The
        # manifest builder already splits/censors such windows (pitfall #61),
        # this guard keeps a hand-edited manifest from repeating the failure.
        if not bool((dt > 0).all()):
            raise ValueError(
                f"window {record['run']}/segment {record['segment']} "
                f"start {record['start']} contains "
                f"{int((~(dt > 0)).sum())} non-positive dt interval(s); "
                "the manifest is not time-monotonic"
            )
        source = store.source[frames]
        source_available = store.source_available[frames]
        if self.with_shards:
            shard_values, shard_mask = _pad_shards(frames, store)
        else:
            # Phase-3 (field-only) windows do not use shard tensors; skipping the
            # per-window padding keeps the CPU side out of the GPU's way.
            shard_values = torch.zeros(len(frames), 0, len(SHARD_FIELD_NAMES))
            shard_mask = torch.zeros(len(frames), 0, dtype=torch.bool)

        u = u_raw
        source_out = source
        if self.normalize:
            if self.stats is None:
                raise ValueError("normalize=True requires stats (use compute_transform_stats)")
            field_mean = torch.as_tensor(self.stats["field_mean"], dtype=torch.float32)
            field_std = torch.as_tensor(self.stats["field_std"], dtype=torch.float32)
            source_mean = torch.as_tensor(self.stats["source_mean"], dtype=torch.float32)
            source_std = torch.as_tensor(self.stats["source_std"], dtype=torch.float32)
            flat = transform_state(u.reshape(-1, NUM_STATE_FIELDS))
            u = standardize(flat, field_mean, field_std, name="u").reshape(u.shape)
            source_out = (
                (source - source_mean) / source_std.clamp_min(1e-12)
            ).to(torch.float32)

        # Frame caches written before ``vertex``/``size``/``x_dofs`` were added
        # carry a topology with those fields set to None; anything that needs the
        # mesh (field-slice plots, FEM operators) silently got None.  Reload from
        # the run's first frame whenever the mesh connectivity is missing.
        if store.topology is None or store.topology.vertex is None:
            store.topology = load_run_topology(self.data_root / record["run"])
        u_deriv = store.derivative_features()[frames]
        if self.normalize and self.stats is not None and "deriv_mean" in self.stats:
            # Physical-unit gradients have wildly different scales from the
            # z-scored value channels; standardise them with train-split stats
            # (see pitfalls #22).
            deriv_mean = torch.as_tensor(self.stats["deriv_mean"], dtype=torch.float32)
            deriv_std = torch.as_tensor(self.stats["deriv_std"], dtype=torch.float32).clamp_min(1e-12)
            u_deriv = (u_deriv - deriv_mean) / deriv_std
        if self.with_lap:
            u_lap = store.laplacian_features()[frames]
        else:
            u_lap = torch.zeros(u.shape[0], u.shape[1], len(FIELD_NAMES))
        if self.with_lap and self.normalize and self.stats is not None and "lap_mean" in self.stats:
            lap_mean = torch.as_tensor(self.stats["lap_mean"], dtype=torch.float32)
            lap_std = torch.as_tensor(self.stats["lap_std"], dtype=torch.float32).clamp_min(1e-12)
            u_lap = (u_lap - lap_mean) / lap_std
        if self.with_condition:
            if self.stats is None or "cond_mean" not in self.stats:
                raise ValueError("with_condition=True requires stats with cond_mean/cond_std")
            cond_mean = torch.as_tensor(self.stats["cond_mean"], dtype=torch.float32)
            cond_std = torch.as_tensor(self.stats["cond_std"], dtype=torch.float32).clamp_min(1e-12)
            condition = (condition - cond_mean) / cond_std
        burn_in = int(record["burn_in"])
        loss_mask = torch.zeros(u.shape[0], dtype=torch.bool)
        loss_mask[burn_in:] = True
        return {
            "run": record["run"],
            "segment": record["segment"],
            "frame_indices": torch.tensor(frames, dtype=torch.long),
            "u_raw": u_raw,
            "u": u,
            "u_deriv": u_deriv,
            "u_lap": u_lap,
            "t": time,
            "dt": dt,
            "condition": condition,
            "source": source_out,
            "source_available": source_available,
            "shard_values": shard_values,
            "shard_mask": shard_mask,
            "loss_mask": loss_mask,
            "topology": store.topology,
        }


def collate_windows(items: Sequence[dict]) -> dict:
    """Stack a batch of windows; the mesh must be shared across the batch."""
    if not items:
        raise ValueError("collate_windows received an empty batch")
    num_frames = items[0]["u"].shape[0]
    num_nodes = items[0]["u"].shape[1]
    for item in items:
        if item["u"].shape[0] != num_frames or item["u"].shape[1] != num_nodes:
            raise ValueError("all windows in a batch must share [T, nodes]")

    topology = items[0]["topology"]
    for item in items[1:]:
        other = item["topology"]
        if not (
            torch.equal(topology.pos, other.pos)
            and torch.equal(topology.edge_index, other.edge_index)
        ):
            raise ValueError("batch mixes different meshes; this is not allowed")

    max_shards = max(item["shard_values"].shape[1] for item in items)
    batch_size, num_frames = len(items), items[0]["u"].shape[0]
    shard_values = torch.zeros(
        batch_size, num_frames, max_shards, items[0]["shard_values"].shape[2]
    )
    shard_mask = torch.zeros(batch_size, num_frames, max_shards, dtype=torch.bool)
    for row, item in enumerate(items):
        count = item["shard_values"].shape[1]
        shard_values[row, :, :count] = item["shard_values"]
        shard_mask[row, :, :count] = item["shard_mask"]

    return {
        "u_raw": torch.stack([item["u_raw"] for item in items]),
        "u": torch.stack([item["u"] for item in items]),
        "u_deriv": torch.stack([item["u_deriv"] for item in items]),
        "u_lap": torch.stack([item["u_lap"] for item in items]),
        "t": torch.stack([item["t"] for item in items]),
        "dt": torch.stack([item["dt"] for item in items]),
        "condition": torch.stack([item["condition"] for item in items]),
        "source": torch.stack([item["source"] for item in items]),
        "source_available": torch.stack([item["source_available"] for item in items]),
        "shard_values": shard_values,
        "shard_mask": shard_mask,
        "loss_mask": torch.stack([item["loss_mask"] for item in items]),
        "runs": [item["run"] for item in items],
        "frame_indices": [item["frame_indices"] for item in items],
        "topology": topology,
    }


def compute_transform_stats(
    stores: dict[str, RunFrameStore] | Sequence[RunFrameStore],
    runs: Iterable[str] | None = None,
    *,
    with_lap: bool = False,
    with_condition: bool = False,
) -> dict:
    """Train-only statistics of the transformed state and the raw source fields.

    ``stores`` may be a mapping run -> store or a sequence of stores.  Stats are
    accumulated over every frame of the selected runs (windows overlap, so this
    is an upper bound on the window-weighted statistics; the caller is
    responsible for restricting to training runs).
    """
    if isinstance(stores, dict):
        selected = [stores[name] for name in (runs or stores.keys())]
    else:
        selected = list(stores)
    count = 0
    frame_count = 0
    field_sum = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    field_sumsq = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    source_sum = np.zeros(NUM_SOURCE_FIELDS, dtype=np.float64)
    source_sumsq = np.zeros(NUM_SOURCE_FIELDS, dtype=np.float64)
    source_count = 0
    deriv_sum = np.zeros(2 * NUM_STATE_FIELDS, dtype=np.float64)
    deriv_sumsq = np.zeros(2 * NUM_STATE_FIELDS, dtype=np.float64)
    deriv_count = 0
    # Frame-to-frame increments of the *transformed* state.  A per-field MSE in
    # z-score space weights every field by its spatial spread, which is the
    # wrong metric for an interpolation model: the increments differ by more
    # than an order of magnitude between channels (psi moves 1.4%/frame, omega
    # 21%), so the loss is dominated by a few channels (pitfalls #30).
    inc_sum = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    inc_sumsq = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    inc_count = 0
    lap_sum = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    lap_sumsq = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    lap_count = 0
    cond_sum = np.zeros(CONDITION_DIM, dtype=np.float64)
    cond_sumsq = np.zeros(CONDITION_DIM, dtype=np.float64)
    cond_count = 0
    # ``transform_state`` clamps the 4 positive fields at 0 before log1p, so any
    # negative raw value (JOREK numerical noise) is *not* recoverable by
    # ``inverse_transform_state``.  Track how often that happens instead of
    # letting it pass silently (tests/test_normalization_contract.py enforces a
    # bound on it).
    negative_counts = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    for store in selected:
        store.load()
        if store.u is None or store.source is None or store.source_available is None:
            raise RuntimeError("frame store is not loaded")
        transformed = transform_state(store.u.reshape(-1, NUM_STATE_FIELDS).float())
        raw_flat = store.u.reshape(-1, NUM_STATE_FIELDS)
        for index in LOG1P_FIELD_INDICES:
            # Only the log1p fields are clamped; the signed fields may be
            # negative by construction.
            negative_counts[index] += float((raw_flat[:, index] < 0).double().sum())
        field_sum += transformed.double().sum(dim=0).numpy()
        field_sumsq += transformed.double().square().sum(dim=0).numpy()
        count += transformed.shape[0]
        if transformed.shape[0] > 1:
            sequence = transformed.reshape(
                store.u.shape[0], -1, NUM_STATE_FIELDS
            ).double()
            increments = (sequence[1:] - sequence[:-1]).reshape(-1, NUM_STATE_FIELDS)
            inc_sum += increments.sum(dim=0).numpy()
            inc_sumsq += increments.square().sum(dim=0).numpy()
            inc_count += increments.shape[0]
        if store.u_dofs is not None:
            deriv = store.derivative_features().reshape(
                -1, 2 * NUM_STATE_FIELDS
            ).double()
            deriv_sum += deriv.sum(dim=0).numpy()
            deriv_sumsq += deriv.square().sum(dim=0).numpy()
            deriv_count += deriv.shape[0]
        if with_lap:
            lap = store.laplacian_features().reshape(-1, NUM_STATE_FIELDS).double()
            lap_sum += lap.sum(dim=0).numpy()
            lap_sumsq += lap.square().sum(dim=0).numpy()
            lap_count += lap.shape[0]
        if with_condition:
            if store.condition is None:
                raise RuntimeError("with_condition=True requires store.condition")
            cond = store.condition.reshape(-1, CONDITION_DIM).double()
            cond_sum += cond.sum(dim=0).numpy()
            cond_sumsq += cond.square().sum(dim=0).numpy()
            cond_count += cond.shape[0]
        frame_count += len(store.frames)
        available = store.source_available
        if bool(available.any()):
            src = store.source[available].reshape(-1, NUM_SOURCE_FIELDS).double()
            source_sum += src.sum(dim=0).numpy()
            source_sumsq += src.square().sum(dim=0).numpy()
            source_count += src.shape[0]
    if count == 0:
        raise ValueError("no state frames selected")
    field_mean = field_sum / count
    field_var = np.clip(field_sumsq / count - np.square(field_mean), 0.0, None)
    field_std_array = np.sqrt(field_var)
    if inc_count:
        inc_mean = inc_sum / inc_count
        # Var(delta) in *normalised* units: the trained target lives there.
        increment_var = np.clip(
            inc_sumsq / inc_count - np.square(inc_mean), 0.0, None
        ) / np.square(np.clip(field_std_array, 1e-12, None))
    else:
        increment_var = np.ones(NUM_STATE_FIELDS)
    if source_count == 0:
        source_mean = np.zeros(NUM_SOURCE_FIELDS)
        source_std = np.ones(NUM_SOURCE_FIELDS)
    else:
        source_mean = source_sum / source_count
        source_var = np.clip(source_sumsq / source_count - np.square(source_mean), 0.0, None)
        source_std = np.sqrt(source_var)
    if cond_count == 0:
        cond_mean = np.zeros(CONDITION_DIM)
        cond_std = np.ones(CONDITION_DIM)
    else:
        cond_mean = cond_sum / cond_count
        cond_var = np.clip(cond_sumsq / cond_count - np.square(cond_mean), 0.0, None)
        # Zero-variance channels (constant namelist entries) must not divide by 0.
        cond_std = np.where(cond_var > 0.0, np.sqrt(cond_var), 1.0)
    return {
        "n_frames": int(frame_count),
        "n_state_values": int(count),
        "n_source_values": int(source_count),
        "field_names": list(FIELD_NAMES),
        "source_names": list(MARKER_PROJECTION_FIELD_NAMES),
        "field_mean": field_mean.tolist(),
        "field_std": field_std_array.tolist(),
        # E[(delta u_norm)^2] per field over the training runs.
        "increment_var": increment_var.tolist(),
        # Fraction of raw values that the log1p clamp zeroes (non-invertible).
        "clamped_fraction": (negative_counts / max(count, 1)).tolist(),
        "source_mean": source_mean.tolist(),
        "source_std": source_std.tolist(),
        "deriv_mean": (deriv_sum / max(deriv_count, 1)).tolist(),
        "deriv_std": np.sqrt(
            np.clip(deriv_sumsq / max(deriv_count, 1) - (deriv_sum / max(deriv_count, 1)) ** 2, 0.0, None)
        ).tolist(),
        "lap_mean": (lap_sum / max(lap_count, 1)).tolist(),
        "lap_std": np.sqrt(
            np.clip(lap_sumsq / max(lap_count, 1) - (lap_sum / max(lap_count, 1)) ** 2, 0.0, None)
        ).tolist(),
        "cond_mean": cond_mean.tolist(),
        "cond_std": cond_std.tolist(),
    }


def to_device(batch: dict, device: torch.device | str) -> dict:
    """Move every tensor of a collated batch to ``device`` (topology included)."""
    device = torch.device(device)
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    topology = batch.get("topology")
    if topology is not None:
        # Keep *all* mesh fields: the previous version rebuilt the topology with
        # only pos/edges/boundary, so ``vertex``/``size``/``x_dofs`` silently
        # became None for every caller that goes through ``to_device`` (field
        # slice plots, FEM assembly).
        def _move(value):
            return None if value is None else value.to(device)

        moved["topology"] = RunTopology(
            run=topology.run,
            pos=topology.pos.to(device),
            edge_index=topology.edge_index.to(device),
            edge_attr=topology.edge_attr.to(device),
            boundary=topology.boundary.to(device),
            size=_move(topology.size),
            vertex=_move(topology.vertex),
            x_dofs=_move(topology.x_dofs),
        )
    return moved


__all__ = [
    "InterpolationWindowDataset",
    "RunFrameStore",
    "RunTopology",
    "collate_windows",
    "compute_transform_stats",
    "load_run_topology",
    "read_frame_payload",
    "to_device",
]
