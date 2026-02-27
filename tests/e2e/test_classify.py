"""E2E tests for classification endpoints.

Endpoints covered:
  POST /classify              (single paper → category)
  POST /abbreviate            (title → short abbreviation)
  POST /papers/classify       (cluster all papers + LLM name)
  POST /categories/rebalance  (alias for /papers/classify)

The /papers/classify test also verifies the float32 serialization bug is fixed.
"""
import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, BATCH_TIMEOUT


class TestClassifySingle:
    def test_classify_paper(self, backend_available, llm_available):
        resp = requests.post(
            f"{BACKEND_URL}/classify",
            json={
                "title": "Attention Is All You Need",
                "abstract": (
                    "We propose a new simple network architecture, the Transformer, "
                    "based solely on attention mechanisms."
                ),
                "existing_categories": ["Computer Vision", "NLP", "Reinforcement Learning"],
            },
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Classify failed: {resp.text}"
        data = resp.json()
        assert "category" in data, f"Missing 'category' in response: {data}"
        assert len(data["category"]) > 0, "Category name is empty"

    def test_classify_missing_title(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/classify",
            json={"abstract": "Some abstract text"},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing title, got {resp.status_code}: {resp.text}"
        )


class TestAbbreviate:
    def test_abbreviate_title(self, backend_available, llm_available):
        resp = requests.post(
            f"{BACKEND_URL}/abbreviate",
            json={"title": "Attention Is All You Need"},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Abbreviate failed: {resp.text}"
        data = resp.json()
        assert "abbreviation" in data, f"Missing 'abbreviation' in response: {data}"
        assert len(data["abbreviation"]) > 0, "Abbreviation is empty"

    def test_abbreviate_missing_title(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/abbreviate",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing title, got {resp.status_code}: {resp.text}"
        )


class TestPapersClassify:
    def test_papers_classify_no_float32_crash(self, backend_available, llm_available):
        """POST /papers/classify must return 200 — specifically, must NOT crash with
        'TypeError: Object of type float32 is not JSON serializable' (BUG-009).

        The fix in clustering.py:603 adds a custom JSON encoder for numpy types.
        """
        resp = requests.post(
            f"{BACKEND_URL}/papers/classify",
            json={},
            timeout=BATCH_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"POST /papers/classify failed (may be float32 bug): "
            f"{resp.status_code}: {resp.text}"
        )
        data = resp.json()
        # Response should include classification statistics
        assert "papers_classified" in data or "clusters_created" in data or len(data) > 0, (
            f"Classify response missing expected fields: {data}"
        )

        # Verify tree was rebuilt with valid structure
        tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=MEDIUM_TIMEOUT)
        assert tree_resp.status_code == 200
        tree = tree_resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree missing 'name' field"


class TestCategoriesRebalance:
    def test_categories_rebalance(self, backend_available, llm_available):
        """POST /categories/rebalance is an alias for /papers/classify."""
        resp = requests.post(
            f"{BACKEND_URL}/categories/rebalance",
            json={},
            timeout=BATCH_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Categories rebalance failed: {resp.status_code}: {resp.text}"
        )
