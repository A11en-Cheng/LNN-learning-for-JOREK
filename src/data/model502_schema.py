"""Authoritative data contract for JOREK Model 502.

This module is intentionally small and dependency-light.  It is the single
source of truth for field order, HDF5 ``values`` indices, transforms and the
first forcing interface.  Model code must import these constants instead of
repeating numeric channel indices.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


SCHEMA_VERSION = "model502-state-v1"
MODEL_ID = 502

# Model 502 Fortran ``mod_model_settings.f90`` uses one-based indices:
# psi=1, u=2, zj=3, w=4, rho=5, Ti=6, Vpar=7, rhoimp=8, Te=9.
FIELD_NAMES: tuple[str, ...] = (
    "psi", "Phi", "Z_j", "omega", "rho", "T_i", "v_par", "rho_imp", "T_e",
)
FIELD_INDEX = {name: index for index, name in enumerate(FIELD_NAMES)}
NUM_STATE_FIELDS = len(FIELD_NAMES)

# HDF5 ``values`` follows the legacy Fortran order above.  The model tensor is
# deliberately reordered to the canonical FIELD_NAMES order.
H5_VALUES_INDICES: tuple[int, ...] = (2, 1, 0, 3, 4, 5, 6, 7, 8)

LOG1P_FIELD_NAMES: tuple[str, ...] = ("rho", "T_i", "rho_imp", "T_e")
LOG1P_FIELD_INDICES: tuple[int, ...] = tuple(FIELD_INDEX[name] for name in LOG1P_FIELD_NAMES)
LINEAR_FIELD_NAMES: tuple[str, ...] = tuple(name for name in FIELD_NAMES if name not in LOG1P_FIELD_NAMES)

# Marker-spi particle projections (``projections*.h5``, use_marker mode) are the
# primary source-control channel (ADR-0005): they are controls projected onto
# the state mesh, never learning targets.
MARKER_PROJECTION_FIELD_NAMES: tuple[str, ...] = (
    "dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp",
)
MARKER_PROJECTION_INDEX = {name: index for index, name in enumerate(MARKER_PROJECTION_FIELD_NAMES)}
NUM_MARKER_PROJECTION_FIELDS = len(MARKER_PROJECTION_FIELD_NAMES)

# Model 502 ``mod_sources:sources`` background controls form an optional second
# channel.  A partial record is an error; nothing is ever silently zero-filled.
BACKGROUND_SOURCE_FIELD_NAMES: tuple[str, ...] = (
    "particle_source", "heat_i_source", "heat_e_source",
)
BACKGROUND_SOURCE_INDEX = {name: index for index, name in enumerate(BACKGROUND_SOURCE_FIELD_NAMES)}
NUM_BACKGROUND_SOURCE_FIELDS = len(BACKGROUND_SOURCE_FIELD_NAMES)
BACKGROUND_SOURCE_DATASET_ALIASES: Mapping[str, tuple[str, ...]] = {
    "particle_source": ("s_rho", "particle_source", "particlesource"),
    "heat_i_source": ("s_t_i", "s_ti", "heat_i_source", "heatsource_i"),
    "heat_e_source": ("s_t_e", "s_te", "heat_e_source", "heatsource_e"),
}
BACKGROUND_SOURCE_VECTOR_DATASET_ALIASES: tuple[str, ...] = (
    "model502_sources", "sources", "source", "controls", "forcing",
)

# Protocol-v1 aliases for the three-channel background control path.  The
# unforced-baseline trainer/processor still speak this vocabulary; they switch
# to the projection channel in #0010.
SOURCE_FIELD_NAMES = BACKGROUND_SOURCE_FIELD_NAMES
SOURCE_FIELD_INDEX = BACKGROUND_SOURCE_INDEX
NUM_SOURCE_FIELDS = NUM_BACKGROUND_SOURCE_FIELDS
SOURCE_DATASET_ALIASES = BACKGROUND_SOURCE_DATASET_ALIASES
SOURCE_VECTOR_DATASET_ALIASES = BACKGROUND_SOURCE_VECTOR_DATASET_ALIASES

CONDITION_FIELD_NAMES: tuple[str, ...] = ("eta", "visco", "ZK_perp", "ZK_par")
NUM_CONDITION_FIELDS = len(CONDITION_FIELD_NAMES)


def validate_feature_tensor(
    value: torch.Tensor,
    names: Sequence[str],
    *,
    name: str,
) -> torch.Tensor:
    """Check the shared ``[nodes, channels]`` tensor contract."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [nodes, {len(names)}], got {tuple(value.shape)}")
    if value.shape[1] != len(names):
        raise ValueError(
            f"{name} must have {len(names)} channels {tuple(names)}, got {value.shape[1]}"
        )
    return value


def validate_state_tensor(value: torch.Tensor, *, name: str = "state") -> torch.Tensor:
    return validate_feature_tensor(value, FIELD_NAMES, name=name)


def validate_source_tensor(value: torch.Tensor, *, name: str = "source") -> torch.Tensor:
    return validate_feature_tensor(value, SOURCE_FIELD_NAMES, name=name)


def validate_projection_tensor(value: torch.Tensor, *, name: str = "marker projection") -> torch.Tensor:
    return validate_feature_tensor(value, MARKER_PROJECTION_FIELD_NAMES, name=name)


def validate_background_source_tensor(
    value: torch.Tensor, *, name: str = "background source"
) -> torch.Tensor:
    return validate_feature_tensor(value, BACKGROUND_SOURCE_FIELD_NAMES, name=name)


def reorder_h5_values(values: torch.Tensor) -> torch.Tensor:
    """Reorder HDF5 values in Fortran order into the canonical state order."""
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError("HDF5 values must have shape [raw_channels, nodes]")
    if values.shape[0] < NUM_STATE_FIELDS:
        raise ValueError(
            f"Model 502 needs at least {NUM_STATE_FIELDS} HDF5 values channels, got {values.shape[0]}"
        )
    return values[list(H5_VALUES_INDICES), :].transpose(0, 1).contiguous()


def transform_state(state: torch.Tensor) -> torch.Tensor:
    """Apply the pre-Z-score Model 502 transform.

    Positive fields use ``log1p(clamp(x, min=0))``.  The clamp is an explicit
    data-quality guard only; callers should record/inspect negative raw values
    rather than treating this as a physics correction.
    """
    state = validate_state_tensor(state)
    transformed = state.clone()
    for index in LOG1P_FIELD_INDICES:
        transformed[:, index] = torch.log1p(torch.clamp(transformed[:, index], min=0.0))
    return transformed


def inverse_transform_state(state: torch.Tensor) -> torch.Tensor:
    """Reverse :func:`transform_state` after de-standardisation."""
    state = validate_state_tensor(state)
    restored = state.clone()
    for index in LOG1P_FIELD_INDICES:
        restored[:, index] = torch.expm1(restored[:, index])
    return restored


def standardize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, name: str) -> torch.Tensor:
    """Z-score an already transformed state/control tensor with checked stats."""
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2 before standardisation")
    mean = torch.as_tensor(mean, device=value.device, dtype=value.dtype).view(-1)
    std = torch.as_tensor(std, device=value.device, dtype=value.dtype).view(-1)
    if mean.numel() != value.shape[1] or std.numel() != value.shape[1]:
        raise ValueError(f"{name} stats must each have {value.shape[1]} entries")
    if torch.any(std <= 0):
        raise ValueError(f"{name} standard deviation must be strictly positive")
    return (value - mean) / std


def destandardize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, name: str) -> torch.Tensor:
    """Inverse :func:`standardize` with the same shape checks."""
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2 before de-standardisation")
    mean = torch.as_tensor(mean, device=value.device, dtype=value.dtype).view(-1)
    std = torch.as_tensor(std, device=value.device, dtype=value.dtype).view(-1)
    if mean.numel() != value.shape[1] or std.numel() != value.shape[1]:
        raise ValueError(f"{name} stats must each have {value.shape[1]} entries")
    return value * std + mean
