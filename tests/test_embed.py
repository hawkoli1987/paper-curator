"""Test embedding endpoint."""
import os

import pytest


@pytest.mark.external
def test_embed(client):
    """Test embedding generation.
    
    Requires external embedding endpoint configured in config/paperqa.yaml.
    Set REQUIRE_EXTERNAL_ENDPOINTS=1 to fail instead of skip when unavailable.
    """
    response = client.post("/embed", json={"text": "This is a test sentence for embedding."})
    if response.status_code == 502:
        if os.environ.get("REQUIRE_EXTERNAL_ENDPOINTS"):
            pytest.fail("Embedding endpoint not available (REQUIRE_EXTERNAL_ENDPOINTS=1)")
        pytest.skip("Embedding endpoint not available")
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "model" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
