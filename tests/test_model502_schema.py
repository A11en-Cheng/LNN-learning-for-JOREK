import pytest

torch = pytest.importorskip("torch")

from src.data.model502_schema import (
    FIELD_NAMES,
    H5_VALUES_INDICES,
    MARKER_PROJECTION_FIELD_NAMES,
    NUM_BACKGROUND_SOURCE_FIELDS,
    NUM_MARKER_PROJECTION_FIELDS,
    NUM_SOURCE_FIELDS,
    validate_background_source_tensor,
    validate_projection_tensor,
    inverse_transform_state,
    reorder_h5_values,
    transform_state,
    validate_source_tensor,
)


def test_h5_values_are_reordered_into_canonical_model502_order():
    raw = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)
    state = reorder_h5_values(raw)
    assert state.shape == (2, len(FIELD_NAMES))
    for output_index, h5_index in enumerate(H5_VALUES_INDICES):
        assert torch.equal(state[:, output_index], raw[h5_index, :])


def test_h5_values_follow_the_fortran_model_order_without_permutation():
    """Regression for the psi/Z_j swap (progress log, 2026-09-29).

    ``mod_export_restart.f90`` copies ``node%values`` into the HDF5 dataset
    without reordering, and Model 502 declares psi=1 / u=2 / Z_j=3 (1-based).
    Any non-identity permutation here silently corrupts the state contract.
    """
    assert H5_VALUES_INDICES == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert FIELD_NAMES[0] == "psi"
    assert FIELD_NAMES[2] == "Z_j"


def test_positive_channel_transform_round_trip():
    state = torch.tensor([
        [1.0, -2.0, 3.0, 4.0, 0.5, 2.0, -1.0, 0.25, 3.0],
        [2.0, -3.0, 4.0, 5.0, 1.5, 4.0, -2.0, 0.75, 6.0],
    ])
    assert torch.allclose(inverse_transform_state(transform_state(state)), state)


def test_source_contract_requires_exactly_three_channels():
    valid = torch.zeros(4, NUM_SOURCE_FIELDS)
    assert validate_source_tensor(valid) is valid
    with pytest.raises(ValueError, match="channels"):
        validate_source_tensor(torch.zeros(4, NUM_SOURCE_FIELDS - 1))


def test_projection_contract_requires_exactly_five_channels():
    assert MARKER_PROJECTION_FIELD_NAMES == (
        "dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp",
    )
    valid = torch.zeros(4, NUM_MARKER_PROJECTION_FIELDS)
    assert validate_projection_tensor(valid) is valid
    with pytest.raises(ValueError, match="channels"):
        validate_projection_tensor(torch.zeros(4, NUM_MARKER_PROJECTION_FIELDS - 1))


def test_background_source_contract_names():
    assert NUM_BACKGROUND_SOURCE_FIELDS == 3
    valid = torch.zeros(4, NUM_BACKGROUND_SOURCE_FIELDS)
    assert validate_background_source_tensor(valid) is valid
