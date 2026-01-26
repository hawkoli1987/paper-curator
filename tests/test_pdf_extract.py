"""Test PDF extraction endpoint."""
import pytest


def test_pdf_extract_file_not_found(client):
    """Test that missing PDF returns 404 error."""
    response = client.post("/pdf/extract", json={"pdf_path": "/nonexistent/file.pdf"})
    assert response.status_code == 404
    assert "PDF not found" in response.json()["detail"]


@pytest.mark.slow
def test_pdf_extract_with_real_pdf(client, sample_arxiv_id, tmp_path):
    """Test PDF extraction with a real downloaded PDF.
    
    Note: This test is marked as slow because it downloads and processes a real PDF.
    """
    # First download a PDF
    download_response = client.post(
        "/arxiv/download",
        json={"arxiv_id": sample_arxiv_id, "output_dir": str(tmp_path)},
    )
    assert download_response.status_code == 200
    pdf_path = download_response.json()["pdf_path"]
    
    # Then extract text from it
    extract_response = client.post("/pdf/extract", json={"pdf_path": pdf_path})
    assert extract_response.status_code == 200
    data = extract_response.json()
    assert "text" in data
    assert "parser" in data
    assert data["parser"] == "paperqa"
    assert len(data["text"]) > 0
