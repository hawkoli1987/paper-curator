"""Test QA endpoint."""
import pytest


@pytest.mark.external
def test_qa_with_context(client):
    """Test QA with provided context.
    
    Note: This test requires external LLM and embedding endpoints.
    """
    context = """
    The Transformer architecture was introduced in the paper "Attention Is All You Need".
    It uses self-attention mechanisms to process sequences in parallel, unlike RNNs which
    process sequentially. The key components are multi-head attention, position-wise 
    feed-forward networks, and positional encodings.
    """
    question = "What are the key components of the Transformer architecture?"
    
    response = client.post("/qa", json={"context": context, "question": question})
    if response.status_code == 502:
        pytest.skip("LLM/Embedding endpoint not available")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


@pytest.mark.external
@pytest.mark.slow
def test_qa_with_pdf(client, sample_arxiv_id, tmp_path):
    """Test QA with PDF path.
    
    Note: This test downloads a real PDF and requires external endpoints.
    """
    # First download a PDF
    download_response = client.post(
        "/arxiv/download",
        json={"arxiv_id": sample_arxiv_id, "output_dir": str(tmp_path)},
    )
    assert download_response.status_code == 200
    pdf_path = download_response.json()["pdf_path"]
    
    # Then ask a question about it
    response = client.post(
        "/qa",
        json={"pdf_path": pdf_path, "question": "What is the main contribution of this paper?"},
    )
    if response.status_code == 502:
        pytest.skip("LLM/Embedding endpoint not available")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_qa_no_input(client):
    """Test that missing context and pdf_path with question still requires input."""
    response = client.post("/qa", json={"question": "What is this about?"})
    # Should return 400 from _paperqa_answer validation
    assert response.status_code in [400, 502]
