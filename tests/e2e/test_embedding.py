"""E2E tests for embedding endpoints.

Endpoints covered:
  POST /embed/abstract   (abstract text → vector)
  POST /embed/fulltext   (full PDF → chunk + index)
"""
import requests
import pytest

from conftest import BACKEND_URL, MEDIUM_TIMEOUT, LLM_TIMEOUT, SAMPLE_ARXIV_ID


class TestEmbedAbstract:
    def test_embed_abstract(self, backend_available, llm_available):
        text = (
            "The Transformer is a model architecture based entirely on attention mechanisms, "
            "dispensing with recurrence and convolutions."
        )
        resp = requests.post(
            f"{BACKEND_URL}/embed/abstract",
            json={"text": text},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Embed abstract failed: {resp.text}"
        data = resp.json()
        assert "embedding" in data, f"Missing 'embedding' in response: {data}"
        embedding = data["embedding"]
        assert isinstance(embedding, list), f"Embedding should be a list, got {type(embedding)}"
        assert len(embedding) > 0, "Embedding vector is empty"
        assert all(isinstance(x, (int, float)) for x in embedding[:10]), (
            "Embedding should contain numeric values"
        )


    def test_embed_missing_text(self, backend_available):
        """Omitting the required 'text' field should return 422."""
        resp = requests.post(
            f"{BACKEND_URL}/embed/abstract",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing text, got {resp.status_code}: {resp.text}"
        )

    def test_embed_dimensions_consistent(self, backend_available, llm_available):
        """Two different texts must produce embeddings of the same dimension."""
        texts = [
            "Attention is all you need.",
            "Reinforcement learning from human feedback.",
        ]
        dims = []
        for text in texts:
            resp = requests.post(
                f"{BACKEND_URL}/embed/abstract",
                json={"text": text},
                timeout=MEDIUM_TIMEOUT,
            )
            assert resp.status_code == 200
            dims.append(len(resp.json()["embedding"]))
        assert dims[0] == dims[1], f"Inconsistent embedding dimensions: {dims}"


class TestEmbedFulltext:
    def test_embed_fulltext(self, backend_available, llm_available, sample_pdf_path):
        """Index full PDF for RAG; verify indexed=True is returned."""
        resp = requests.post(
            f"{BACKEND_URL}/embed/fulltext",
            json={"pdf_path": sample_pdf_path, "arxiv_id": SAMPLE_ARXIV_ID},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Embed fulltext failed: {resp.text}"
        data = resp.json()
        assert "indexed" in data, f"Missing 'indexed' in response: {data}"
        assert data["indexed"] is True, (
            f"Expected indexed=True, got {data['indexed']}"
        )
