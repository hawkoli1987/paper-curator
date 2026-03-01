"""Input validation tests using requests to call the running backend.

These tests verify that endpoints properly validate their inputs and return
appropriate error responses for missing/invalid fields.
"""
import os
import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")


@pytest.fixture(scope="module")
def backend_available():
    """Ensure backend is running."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if resp.status_code != 200:
            pytest.skip("Backend not healthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not available")
    return True


class TestArxivValidation:
    """Validate arXiv endpoint inputs."""

    def test_arxiv_resolve_requires_identifier(self, backend_available):
        """POST /arxiv/resolve requires arxiv_id or url."""
        resp = requests.post(f"{BACKEND_URL}/arxiv/resolve", json={}, timeout=10)
        assert resp.status_code == 400
        assert "arxiv_id" in resp.text.lower() or "url" in resp.text.lower()

    def test_arxiv_download_requires_identifier(self, backend_available):
        """POST /arxiv/download requires arxiv_id or url."""
        resp = requests.post(f"{BACKEND_URL}/arxiv/download", json={}, timeout=10)
        assert resp.status_code == 400
        assert "arxiv_id" in resp.text.lower() or "url" in resp.text.lower()


class TestPdfValidation:
    """Validate PDF endpoint inputs."""

    def test_pdf_extract_requires_path(self, backend_available):
        """POST /pdf/extract requires pdf_path."""
        resp = requests.post(f"{BACKEND_URL}/pdf/extract", json={}, timeout=10)
        assert resp.status_code == 422


class TestSummarizeValidation:
    """Validate summarize endpoint inputs."""

    def test_summarize_structured_requires_at_least_one_field(self, backend_available):
        """POST /summarize/structured with no fields returns 422 (neither pdf_path nor arxiv_id)."""
        resp = requests.post(f"{BACKEND_URL}/summarize/structured", json={}, timeout=10)
        assert resp.status_code == 422

    def test_summarize_structured_accepts_arxiv_id_alone(self, backend_available):
        """POST /summarize/structured with only arxiv_id is accepted (pdf_path is now optional)."""
        resp = requests.post(
            f"{BACKEND_URL}/summarize/structured",
            json={"arxiv_id": "1706.03762"},
            timeout=600,
        )
        # Should NOT return 422 — the model no longer requires pdf_path
        assert resp.status_code != 422, f"Got 422: pdf_path is still required but should be optional"


class TestEmbedValidation:
    """Validate embedding endpoint inputs."""

    def test_embed_query_requires_text(self, backend_available):
        """POST /embed_query requires text."""
        resp = requests.post(f"{BACKEND_URL}/embed_query", json={}, timeout=10)
        assert resp.status_code == 422

    def test_embed_abstract_requires_text(self, backend_available):
        """POST /embed/abstract requires text."""
        resp = requests.post(f"{BACKEND_URL}/embed/abstract", json={}, timeout=10)
        assert resp.status_code == 422

    def test_embed_fulltext_requires_fields(self, backend_available):
        """POST /embed/fulltext requires arxiv_id and pdf_path."""
        resp = requests.post(f"{BACKEND_URL}/embed/fulltext", json={}, timeout=10)
        assert resp.status_code == 422


class TestQAValidation:
    """Validate Q&A endpoint inputs."""

    def test_qa_requires_question(self, backend_available):
        """POST /qa requires question."""
        resp = requests.post(f"{BACKEND_URL}/qa", json={}, timeout=10)
        assert resp.status_code == 422

    def test_qa_structured_requires_arxiv_id(self, backend_available):
        """POST /qa/structured requires arxiv_id."""
        resp = requests.post(f"{BACKEND_URL}/qa/structured", json={}, timeout=10)
        assert resp.status_code == 422


class TestClassifyValidation:
    """Validate classification endpoint inputs."""

    def test_abbreviate_requires_title(self, backend_available):
        """POST /abbreviate requires title."""
        resp = requests.post(f"{BACKEND_URL}/abbreviate", json={}, timeout=10)
        assert resp.status_code == 422


class TestPaperValidation:
    """Validate paper endpoint inputs."""

    def test_save_paper_requires_fields(self, backend_available):
        """POST /papers/save requires arxiv_id, title, and authors."""
        resp = requests.post(f"{BACKEND_URL}/papers/save", json={}, timeout=10)
        assert resp.status_code == 422

    def test_reabbreviate_requires_arxiv_id(self, backend_available):
        """POST /papers/reabbreviate requires arxiv_id."""
        resp = requests.post(f"{BACKEND_URL}/papers/reabbreviate", json={}, timeout=10)
        assert resp.status_code == 422


class TestTopicValidation:
    """Validate topic endpoint inputs."""

    def test_topic_search_requires_topic(self, backend_available):
        """POST /topic/search requires topic field."""
        resp = requests.post(f"{BACKEND_URL}/topic/search", json={}, timeout=10)
        assert resp.status_code == 422

    def test_topic_create_requires_fields(self, backend_available):
        """POST /topic/create requires name."""
        resp = requests.post(f"{BACKEND_URL}/topic/create", json={}, timeout=10)
        assert resp.status_code == 422


class TestReferencesValidation:
    """Validate references endpoint inputs."""

    def test_references_fetch_requires_arxiv_id(self, backend_available):
        """POST /references/fetch requires arxiv_id."""
        resp = requests.post(f"{BACKEND_URL}/references/fetch", json={}, timeout=10)
        assert resp.status_code == 422

    def test_references_explain_requires_fields(self, backend_available):
        """POST /references/explain requires reference_id, source_paper_title, cited_title."""
        resp = requests.post(f"{BACKEND_URL}/references/explain", json={}, timeout=10)
        assert resp.status_code == 422


class TestSimilarValidation:
    """Validate similar papers endpoint inputs."""

    def test_similar_requires_arxiv_id(self, backend_available):
        """POST /papers/similar requires arxiv_id."""
        resp = requests.post(f"{BACKEND_URL}/papers/similar", json={}, timeout=10)
        assert resp.status_code == 422
