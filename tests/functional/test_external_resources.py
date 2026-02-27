"""Functional tests for external resource operations."""
import os
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")


class TestGitHubRepo:
    """Test GitHub repository fetching."""

    def test_fetch_github_repo(self, backend_available):
        """Fetch GitHub repository metadata for a paper."""
        resp = requests.post(
            f"{BACKEND_URL}/repos/search",
            json={
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need"
            },
            timeout=60
        )
        
        # Endpoint must exist and return valid response
        assert resp.status_code == 200, f"Repos search failed with status {resp.status_code}: {resp.text}"
        data = resp.json()
        
        # Response should contain repos info
        assert isinstance(data, dict)


class TestReferences:
    """Test reference fetching."""

    def test_fetch_references(self, backend_available):
        """Fetch references for a paper."""
        resp = requests.post(
            f"{BACKEND_URL}/references/fetch",
            json={
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need"
            },
            timeout=120
        )

        # Endpoint must work - no skipping on 404/503
        assert resp.status_code == 200, f"References fetch failed with status {resp.status_code}: {resp.text}"
        data = resp.json()

        assert "references" in data or "citations" in data, f"Response missing references field: {list(data.keys())}"
        refs = data.get("references", data.get("citations", []))
        # This paper should have references
        assert isinstance(refs, list)

    def test_explain_reference_cache(self, backend_available):
        """Second call to /references/explain for the same reference_id returns from_cache=True."""
        # First fetch references to get a valid reference_id
        fetch_resp = requests.post(
            f"{BACKEND_URL}/references/fetch",
            json={"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
            timeout=120,
        )
        assert fetch_resp.status_code == 200
        refs = fetch_resp.json().get("references", fetch_resp.json().get("citations", []))
        if not refs:
            import pytest
            pytest.skip("No references found for cache test")

        ref_id = refs[0].get("id")
        if not ref_id:
            import pytest
            pytest.skip("Reference has no id field")

        payload = {
            "reference_id": ref_id,
            "source_paper_title": "Attention Is All You Need",
            "cited_title": refs[0].get("title", "Unknown"),
        }

        # First call — generates explanation
        r1 = requests.post(f"{BACKEND_URL}/references/explain", json=payload, timeout=60)
        assert r1.status_code == 200, f"First explain call failed: {r1.text}"

        # Second call — must hit cache (BUG-004 fix: get_reference_by_id used correctly)
        r2 = requests.post(f"{BACKEND_URL}/references/explain", json=payload, timeout=30)
        assert r2.status_code == 200, f"Second explain call failed: {r2.text}"
        assert r2.json().get("from_cache") is True, (
            f"Cache miss on second call — BUG-004 not fixed. Response: {r2.json()}"
        )


class TestSimilarPapers:
    """Test similar papers functionality."""

    def test_fetch_similar(self, backend_available):
        """Fetch similar papers."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/similar",
            json={"arxiv_id": "1706.03762"},
            timeout=60
        )
        
        # Endpoint must work
        assert resp.status_code == 200, f"Similar papers failed with status {resp.status_code}: {resp.text}"
        data = resp.json()
        
        # API returns 'similar_papers' key
        assert "similar_papers" in data, f"Response missing similar_papers: {list(data.keys())}"
        similar = data["similar_papers"]
        assert isinstance(similar, list)
        # Should find at least one similar paper
        assert len(similar) >= 1, "No similar papers found"
        # Each paper should have expected fields
        paper = similar[0]
        assert "title" in paper, f"Similar paper missing title: {paper}"


class TestTopicOperations:
    """Test topic-based operations."""

    def test_topic_list(self, backend_available):
        """List all topics."""
        resp = requests.get(
            f"{BACKEND_URL}/topic/list",
            timeout=30
        )
        
        # Endpoint must work
        assert resp.status_code == 200, f"Topic list failed with status {resp.status_code}: {resp.text}"
        data = resp.json()
        
        # Response should be a list or contain a topics list
        if isinstance(data, list):
            topics = data
        else:
            topics = data.get("topics", [])
        
        assert isinstance(topics, list)
