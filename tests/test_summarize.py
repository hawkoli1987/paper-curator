"""Test summarization endpoint."""
import os

import pytest


def _check_endpoint_available(response: "Response") -> None:
    """Check if endpoint is available, skip or fail based on env var."""
    if response.status_code == 502:
        if os.environ.get("REQUIRE_EXTERNAL_ENDPOINTS"):
            pytest.fail("LLM/Embedding endpoint not available (REQUIRE_EXTERNAL_ENDPOINTS=1)")
        pytest.skip("LLM/Embedding endpoint not available")


@pytest.mark.external
def test_summarize_with_text(client):
    """Test summarization with provided text.
    
    Requires external LLM and embedding endpoints.
    Set REQUIRE_EXTERNAL_ENDPOINTS=1 to fail instead of skip when unavailable.
    """
    sample_text = """
    Attention mechanisms have become an integral part of compelling sequence modeling 
    and transduction models in various tasks, allowing modeling of dependencies without 
    regard to their distance in the input or output sequences. We propose a new simple 
    network architecture, the Transformer, based solely on attention mechanisms, 
    dispensing with recurrence and convolutions entirely.
    """
    response = client.post("/summarize", json={"text": sample_text})
    _check_endpoint_available(response)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "prompt_id" in data
    assert "model" in data


@pytest.mark.external
@pytest.mark.slow
def test_summarize_with_pdf(client, sample_arxiv_id, tmp_path):
    """Test summarization with PDF path.
    
    Downloads a real PDF and requires external endpoints.
    """
    download_response = client.post(
        "/arxiv/download",
        json={"arxiv_id": sample_arxiv_id, "output_dir": str(tmp_path)},
    )
    assert download_response.status_code == 200
    pdf_path = download_response.json()["pdf_path"]
    
    response = client.post("/summarize", json={"pdf_path": pdf_path})
    _check_endpoint_available(response)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data


def test_summarize_no_input(client):
    """Test that missing input returns error."""
    response = client.post("/summarize", json={})
    # 502 if endpoint unreachable, 500 if assert fails
    assert response.status_code in [500, 502]
