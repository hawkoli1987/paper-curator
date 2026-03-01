"""E2E tests for references, similarity, and repo search endpoints.

Endpoints covered:
  POST /references/fetch
  GET  /references/explain  (or POST, depending on router)
  POST /papers/similar
  POST /repos/search
"""
import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, SAMPLE_ARXIV_ID


class TestFetchReferences:
    def test_fetch_references(self, backend_available, existing_paper):
        arxiv_id = existing_paper["arxiv_id"]
        resp = requests.post(
            f"{BACKEND_URL}/references/fetch",
            json={"arxiv_id": arxiv_id},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"References fetch failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "references" in data, f"Missing 'references' in response: {data}"
        assert isinstance(data["references"], list), (
            f"'references' should be a list: {data['references']}"
        )

    def test_fetch_references_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/references/fetch",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )


class TestExplainReference:
    def test_explain_reference(self, backend_available, llm_available, existing_paper):
        """Fetch references first, then explain the first one."""
        arxiv_id = existing_paper["arxiv_id"]
        fetch_resp = requests.post(
            f"{BACKEND_URL}/references/fetch",
            json={"arxiv_id": arxiv_id},
            timeout=MEDIUM_TIMEOUT,
        )
        assert fetch_resp.status_code == 200
        refs = fetch_resp.json().get("references", [])
        if not refs:
            pytest.skip(f"No references found for paper {arxiv_id}")

        # Take the first reference
        ref = refs[0]
        ref_id = ref.get("paperId") or ref.get("paper_id") or ref.get("id")
        if not ref_id:
            pytest.skip("Reference has no usable ID field")

        explain_resp = requests.post(
            f"{BACKEND_URL}/references/explain",
            json={
                "reference_id": ref.get("id"),
                "source_paper_title": existing_paper.get("title", ""),
                "cited_title": ref.get("cited_title", ""),
                "citation_context": ref.get("citation_context"),
            },
            timeout=LLM_TIMEOUT,
        )
        assert explain_resp.status_code == 200, (
            f"References explain failed: {explain_resp.status_code}: {explain_resp.text}"
        )
        data = explain_resp.json()
        assert "explanation" in data, f"Missing 'explanation' in response: {data}"
        assert len(data["explanation"]) > 10, "Explanation too short"

    def test_explain_reference_cache(self, backend_available, llm_available, existing_paper):
        """Second call to /references/explain for the same pair should be cached."""
        arxiv_id = existing_paper["arxiv_id"]
        fetch_resp = requests.post(
            f"{BACKEND_URL}/references/fetch",
            json={"arxiv_id": arxiv_id},
            timeout=MEDIUM_TIMEOUT,
        )
        assert fetch_resp.status_code == 200
        refs = fetch_resp.json().get("references", [])
        if not refs:
            pytest.skip(f"No references for {arxiv_id}")

        ref = refs[0]
        ref_id = ref.get("paperId") or ref.get("paper_id") or ref.get("id")
        if not ref_id:
            pytest.skip("Reference has no usable ID field")

        payload = {
            "reference_id": ref.get("id"),
            "source_paper_title": existing_paper.get("title", ""),
            "cited_title": ref.get("cited_title", ""),
            "citation_context": ref.get("citation_context"),
        }
        # First call (may or may not be cached)
        requests.post(
            f"{BACKEND_URL}/references/explain",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        # Second call — should be served from cache
        second_resp = requests.post(
            f"{BACKEND_URL}/references/explain",
            json=payload,
            timeout=MEDIUM_TIMEOUT,
        )
        assert second_resp.status_code == 200
        data = second_resp.json()
        assert data.get("from_cache") is True, (
            f"Expected from_cache=True on second call, got: {data}"
        )


class TestSimilarPapers:
    def test_similar_papers(self, backend_available, existing_paper):
        arxiv_id = existing_paper["arxiv_id"]
        resp = requests.post(
            f"{BACKEND_URL}/papers/similar",
            json={"arxiv_id": arxiv_id, "limit": 5},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Similar papers failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "similar_papers" in data or "papers" in data, (
            f"Missing similar papers list in response: {data}"
        )
        results = data.get("similar_papers") or data.get("papers", [])
        assert isinstance(results, list), f"Similar papers should be a list: {results}"

    def test_similar_papers_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/similar",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )


class TestReposSearch:
    def test_repos_search(self, backend_available, existing_paper):
        """POST /repos/search requires arxiv_id and title (not 'query')."""
        resp = requests.post(
            f"{BACKEND_URL}/repos/search",
            json={
                "arxiv_id": existing_paper["arxiv_id"],
                "title": existing_paper.get("title", "Attention Is All You Need"),
            },
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Repos search failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert isinstance(data, dict), (
            f"Repos search response should be dict: {type(data)}"
        )
        assert "repos" in data, f"Missing 'repos' in response: {data}"

    def test_repos_search_missing_required_fields(self, backend_available):
        """RepoSearchRequest requires arxiv_id and title — empty payload returns 422."""
        resp = requests.post(
            f"{BACKEND_URL}/repos/search",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing fields, got {resp.status_code}: {resp.text}"
        )
