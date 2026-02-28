"""E2E tests for classification endpoints.

Endpoints covered:
  POST /abbreviate            (title → short abbreviation)
  POST /papers/categorize     (cluster all papers + LLM name, replaces /papers/classify)
"""
import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, BATCH_TIMEOUT


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


class TestPapersCategorize:
    def test_papers_categorize_partial(self, backend_available, llm_available):
        """POST /papers/categorize (partial) — re-clusters dirty nodes.

        Also verifies the float32 serialization fix (BUG-009) remains in place.
        """
        resp = requests.post(
            f"{BACKEND_URL}/papers/categorize",
            json={},
            timeout=BATCH_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"POST /papers/categorize failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "mode" in data and data["mode"] == "partial", f"Expected partial mode: {data}"
        assert "recategorized" in data or "message" in data, (
            f"Categorize response missing expected fields: {data}"
        )

        # Verify tree has valid structure
        tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=MEDIUM_TIMEOUT)
        assert tree_resp.status_code == 200
        tree = tree_resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree missing 'name' field"

    def test_papers_categorize_full(self, backend_available, llm_available):
        """POST /papers/categorize?full=true — full rebuild."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/categorize?full=true",
            json={},
            timeout=BATCH_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"POST /papers/categorize?full=true failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "mode" in data and data["mode"] == "full", f"Expected full mode: {data}"
