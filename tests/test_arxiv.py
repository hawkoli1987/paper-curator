"""Test arXiv endpoints."""
import pytest


def test_arxiv_resolve_with_id(client, sample_arxiv_id):
    """Test resolving arXiv paper by ID."""
    response = client.post("/arxiv/resolve", json={"arxiv_id": sample_arxiv_id})
    assert response.status_code == 200
    data = response.json()
    assert "arxiv_id" in data
    assert "title" in data
    assert "authors" in data
    assert "summary" in data
    assert "pdf_url" in data
    # Verify it's the "Attention Is All You Need" paper
    assert "Attention" in data["title"]


def test_arxiv_resolve_with_url(client, sample_arxiv_url):
    """Test resolving arXiv paper by URL."""
    response = client.post("/arxiv/resolve", json={"url": sample_arxiv_url})
    assert response.status_code == 200
    data = response.json()
    assert "arxiv_id" in data
    assert "title" in data


def test_arxiv_resolve_no_identifier(client):
    """Test that missing identifier returns 400 error."""
    response = client.post("/arxiv/resolve", json={})
    assert response.status_code == 400
    assert "Provide arxiv_id or url" in response.json()["detail"]


def test_arxiv_resolve_invalid_id(client):
    """Test that invalid arXiv ID returns 404 error."""
    response = client.post("/arxiv/resolve", json={"arxiv_id": "invalid.id.here"})
    assert response.status_code == 404


@pytest.mark.slow
def test_arxiv_download(client, sample_arxiv_id, tmp_path):
    """Test downloading arXiv paper PDF.
    
    Note: This test is marked as slow because it downloads a real PDF.
    """
    response = client.post(
        "/arxiv/download",
        json={"arxiv_id": sample_arxiv_id, "output_dir": str(tmp_path)},
    )
    assert response.status_code == 200
    data = response.json()
    assert "arxiv_id" in data
    assert "pdf_path" in data
    assert data["pdf_path"] is not None
