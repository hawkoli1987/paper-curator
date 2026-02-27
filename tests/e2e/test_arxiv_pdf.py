"""E2E tests for arXiv resolution, PDF download, and text extraction.

Endpoints covered:
  POST /arxiv/resolve
  POST /arxiv/download
  POST /pdf/extract
"""
import os
import shutil
from pathlib import Path

import pytest
import requests

from conftest import BACKEND_URL, MEDIUM_TIMEOUT, SAMPLE_ARXIV_ID


SCRATCH_DIR = str(
    Path(__file__).parent.parent / "storage" / "downloads" / "e2e_scratch"
)


@pytest.fixture(autouse=True)
def scratch_cleanup():
    """Ensure scratch dir is removed after each test in this module."""
    yield
    if Path(SCRATCH_DIR).exists():
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)


class TestArxivResolve:
    def test_resolve_by_id(self, backend_available, sample_arxiv_id):
        resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": sample_arxiv_id},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Resolve failed: {resp.text}"
        data = resp.json()
        assert "title" in data, f"Missing 'title' in response: {data}"
        assert "authors" in data, f"Missing 'authors' in response: {data}"
        assert "summary" in data, f"Missing 'summary' in response: {data}"
        assert "pdf_url" in data, f"Missing 'pdf_url' in response: {data}"
        assert "Attention" in data["title"], (
            f"Expected 'Attention' in title: {data['title']}"
        )
        assert len(data.get("authors", [])) > 0, "Expected non-empty authors list"
        assert len(data.get("summary", "")) > 100, "Abstract too short"

    def test_resolve_invalid_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": "0000.00000"},
            timeout=MEDIUM_TIMEOUT,
        )
        # Invalid IDs should return 404 or 400, not 200
        assert resp.status_code in [400, 404], (
            f"Expected 400/404 for invalid arXiv ID, got {resp.status_code}: {resp.text}"
        )


class TestArxivDownload:
    def test_download_pdf(self, backend_available, sample_arxiv_id):
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        resp = requests.post(
            f"{BACKEND_URL}/arxiv/download",
            json={"arxiv_id": sample_arxiv_id, "output_dir": SCRATCH_DIR},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Download failed: {resp.text}"
        data = resp.json()
        assert "pdf_path" in data, f"Missing 'pdf_path' in response: {data}"
        pdf_path = Path(data["pdf_path"])
        assert pdf_path.exists(), f"PDF not found on disk at {pdf_path}"
        assert pdf_path.stat().st_size > 100_000, (
            f"PDF too small ({pdf_path.stat().st_size} bytes) — likely corrupt"
        )


class TestPdfExtract:
    def test_extract_text(self, backend_available, sample_pdf_path):
        resp = requests.post(
            f"{BACKEND_URL}/pdf/extract",
            json={"pdf_path": sample_pdf_path},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, f"PDF extract failed: {resp.text}"
        data = resp.json()
        assert "text" in data, f"Missing 'text' in response: {data}"
        text = data["text"]
        assert len(text) > 5000, f"Extracted text too short: {len(text)} chars"
        assert "\x00" not in text, "Extracted text contains NUL bytes (will break PostgreSQL)"
        assert "attention" in text.lower(), "Expected 'attention' in extracted text"
        assert "transformer" in text.lower(), "Expected 'transformer' in extracted text"

    def test_extract_missing_path(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/pdf/extract",
            json={"pdf_path": "/nonexistent/path/to/file.pdf"},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code in [400, 404, 422], (
            f"Expected 400/404/422 for missing PDF, got {resp.status_code}: {resp.text}"
        )
