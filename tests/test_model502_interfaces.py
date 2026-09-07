import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.data.model502_schema import NUM_BACKGROUND_SOURCE_FIELDS, NUM_STATE_FIELDS
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor


def test_model502_network_interface_returns_nine_field_residual():
    x = torch.randn(3, NUM_STATE_FIELDS)
    pos = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)

    encoder = JorekGNNEncoder(hidden_dim=16, latent_dim=8, num_layers=2)
    processor = JorekLNNProcessor(latent_dim=8, hidden_size=8)
    decoder = JorekGNNDecoder(latent_dim=8, hidden_dim=16, num_layers=2)

    latent = encoder(x, edge_index, pos)
    next_latent, _ = processor(
        latent, batch, controls=torch.randn(3, NUM_BACKGROUND_SOURCE_FIELDS)
    )
    delta, _ = decoder(next_latent, edge_index, batch, pos)
    assert delta.shape == (3, NUM_STATE_FIELDS)


def test_model502_checkpoint_reload_preserves_output(tmp_path):
    torch.manual_seed(7)
    x = torch.randn(3, NUM_STATE_FIELDS)
    pos = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    controls = torch.randn(3, NUM_BACKGROUND_SOURCE_FIELDS)

    encoder = JorekGNNEncoder(hidden_dim=16, latent_dim=8, num_layers=2).eval()
    processor = JorekLNNProcessor(latent_dim=8, hidden_size=8).eval()
    decoder = JorekGNNDecoder(latent_dim=8, hidden_dim=16, num_layers=2).eval()
    with torch.no_grad():
        latent = encoder(x, edge_index, pos)
        next_latent, _ = processor(latent, batch, controls=controls)
        expected_delta, expected_disruption = decoder(next_latent, edge_index, batch, pos)

    checkpoint_path = tmp_path / "model502.pt"
    torch.save({
        "schema_version": "model502-state-v1",
        "encoder_state_dict": encoder.state_dict(),
        "processor_state_dict": processor.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["schema_version"] == "model502-state-v1"

    restored_encoder = JorekGNNEncoder(hidden_dim=16, latent_dim=8, num_layers=2).eval()
    restored_processor = JorekLNNProcessor(latent_dim=8, hidden_size=8).eval()
    restored_decoder = JorekGNNDecoder(latent_dim=8, hidden_dim=16, num_layers=2).eval()
    restored_encoder.load_state_dict(checkpoint["encoder_state_dict"])
    restored_processor.load_state_dict(checkpoint["processor_state_dict"])
    restored_decoder.load_state_dict(checkpoint["decoder_state_dict"])

    with torch.no_grad():
        latent = restored_encoder(x, edge_index, pos)
        next_latent, _ = restored_processor(latent, batch, controls=controls)
        actual_delta, actual_disruption = restored_decoder(next_latent, edge_index, batch, pos)
    assert torch.allclose(actual_delta, expected_delta, atol=1e-6)
    assert torch.allclose(actual_disruption, expected_disruption, atol=1e-6)
