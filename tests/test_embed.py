"""Test embedding endpoint."""
import pytest


@pytest.mark.external
def test_embed(client):
    """Test embedding generation.
    
    Note: This test requires an external embedding endpoint configured in config/paperqa.yaml.
    Mark as external to skip in CI without proper setup.
    """
    response = client.post("/embed", json={"text": "This is a test sentence for embedding."})
    # If embedding endpoint is not available, we expect a 502 error
    if response.status_code == 502:
        pytest.skip("Embedding endpoint not available")
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "model" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
