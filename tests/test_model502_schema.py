import pytest

torch = pytest.importorskip("torch")

from src.data.model502_schema import (
    FIELD_NAMES,
    H5_VALUES_INDICES,
    NUM_SOURCE_FIELDS,
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
