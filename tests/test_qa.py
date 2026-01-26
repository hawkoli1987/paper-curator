"""Test QA endpoint."""
import os

import pytest


def _check_endpoint_available(response: "Response") -> None:
    """Check if endpoint is available, skip or fail based on env var."""
    if response.status_code == 502:
        if os.environ.get("REQUIRE_EXTERNAL_ENDPOINTS"):
            pytest.fail("LLM/Embedding endpoint not available (REQUIRE_EXTERNAL_ENDPOINTS=1)")
        pytest.skip("LLM/Embedding endpoint not available")


@pytest.mark.external
def test_qa_with_context(client):
    """Test QA with provided context.
    
    Requires external LLM and embedding endpoints.
    Set REQUIRE_EXTERNAL_ENDPOINTS=1 to fail instead of skip when unavailable.
    """
    context = """
    The Transformer architecture was introduced in the paper "Attention Is All You Need".
    It uses self-attention mechanisms to process sequences in parallel, unlike RNNs which
    process sequentially. The key components are multi-head attention, position-wise 
    feed-forward networks, and positional encodings.
    """
    question = "What are the key components of the Transformer architecture?"
    
    response = client.post("/qa", json={"context": context, "question": question})
    _check_endpoint_available(response)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


@pytest.mark.external
@pytest.mark.slow
def test_qa_with_pdf(client, sample_arxiv_id, tmp_path):
    """Test QA with PDF path.
    
    Downloads a real PDF and requires external endpoints.
    """
    download_response = client.post(
        "/arxiv/download",
        json={"arxiv_id": sample_arxiv_id, "output_dir": str(tmp_path)},
    )
    assert download_response.status_code == 200
    pdf_path = download_response.json()["pdf_path"]
    
    response = client.post(
        "/qa",
        json={"pdf_path": pdf_path, "question": "What is the main contribution of this paper?"},
    )
    _check_endpoint_available(response)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_qa_no_input(client):
    """Test that missing context and pdf_path returns error."""
    response = client.post("/qa", json={"question": "What is this about?"})
    # 502 if endpoint unreachable, 500 if assert fails
    assert response.status_code in [500, 502]
