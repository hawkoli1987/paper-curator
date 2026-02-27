"""E2E tests for all topic endpoints.

Endpoints covered:
  POST   /topic/search
  POST   /topic/create
  GET    /topic/list
  GET    /topic/check
  GET    /topic/{topic_id}
  POST   /topic/{topic_id}/papers
  DELETE /topic/{topic_id}/papers/{paper_id}
  POST   /topic/{topic_id}/query
  DELETE /topic/{topic_id}
"""
import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, SAMPLE_ARXIV_ID

TOPIC_NAME = "e2e_test_transformer_attention"
TOPIC_QUERY = "transformer attention mechanisms"


@pytest.fixture(scope="module")
def test_topic(backend_available, llm_available):
    """Create a topic for the module, delete it on teardown.

    Requires llm_available because /topic/create calls the embedding API.
    """
    create_resp = requests.post(
        f"{BACKEND_URL}/topic/create",
        json={"name": TOPIC_NAME, "topic_query": TOPIC_QUERY},
        timeout=LLM_TIMEOUT,
    )
    assert create_resp.status_code == 200, (
        f"Topic creation for fixture failed: {create_resp.status_code}: {create_resp.text}"
    )
    data = create_resp.json()
    topic_id = data.get("topic_id") or data.get("id")
    assert topic_id is not None, f"Could not extract topic_id from: {data}"
    yield topic_id
    # Cleanup
    requests.delete(f"{BACKEND_URL}/topic/{topic_id}", timeout=MEDIUM_TIMEOUT)


class TestTopicSearch:
    def test_topic_search(self, backend_available, llm_available):
        resp = requests.post(
            f"{BACKEND_URL}/topic/search",
            json={"topic": TOPIC_QUERY, "limit": 10, "offset": 0},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Topic search failed: {resp.text}"
        data = resp.json()
        assert "papers" in data, f"Missing 'papers' in response: {data}"
        assert isinstance(data["papers"], list), "'papers' should be a list"

    def test_topic_search_missing_topic(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/topic/search",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing topic, got {resp.status_code}: {resp.text}"
        )


class TestTopicList:
    def test_topic_list(self, backend_available):
        """GET /topic/list returns {"topics": [...]} not a bare list."""
        resp = requests.get(f"{BACKEND_URL}/topic/list", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200, f"Topic list failed: {resp.text}"
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"
        assert "topics" in data, f"Missing 'topics' key in response: {data}"
        assert isinstance(data["topics"], list), (
            f"topics should be a list, got {type(data['topics'])}"
        )


class TestTopicCheck:
    def test_topic_check_exists(self, backend_available, test_topic):
        """GET /topic/check uses 'topic_query' (not 'name') as query parameter."""
        resp = requests.get(
            f"{BACKEND_URL}/topic/check",
            params={"topic_query": TOPIC_QUERY},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Topic check failed: {resp.text}"
        data = resp.json()
        assert "exists" in data, f"Missing 'exists' in response: {data}"
        assert data["exists"] is True, (
            f"Expected exists=True for '{TOPIC_QUERY}', got: {data}"
        )

    def test_topic_check_not_exists(self, backend_available):
        """Check for a query that definitely doesn't exist."""
        resp = requests.get(
            f"{BACKEND_URL}/topic/check",
            params={"topic_query": "e2e_topic_query_that_does_not_exist_xyz_99999"},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("exists") is False, (
            f"Expected exists=False for nonexistent topic query: {data}"
        )


class TestTopicGet:
    def test_get_existing_topic(self, backend_available, test_topic):
        resp = requests.get(
            f"{BACKEND_URL}/topic/{test_topic}",
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"GET /topic/{test_topic} failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"

    def test_get_nonexistent_topic(self, backend_available):
        resp = requests.get(
            f"{BACKEND_URL}/topic/999999999",
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 404, (
            f"Expected 404 for nonexistent topic, got {resp.status_code}: {resp.text}"
        )


class TestTopicPapers:
    def test_add_and_remove_paper(self, backend_available, test_topic, existing_paper):
        """Add a paper to the topic, verify, then remove it."""
        paper_id = existing_paper.get("paper_id") or existing_paper.get("arxiv_id")
        assert paper_id, f"Could not get paper_id from existing_paper: {existing_paper}"

        # Add paper
        add_resp = requests.post(
            f"{BACKEND_URL}/topic/{test_topic}/papers",
            json={"paper_ids": [paper_id], "similarity_scores": [0.9]},
            timeout=MEDIUM_TIMEOUT,
        )
        assert add_resp.status_code == 200, (
            f"Add paper to topic failed: {add_resp.status_code}: {add_resp.text}"
        )
        data = add_resp.json()
        assert "added" in data, f"Missing 'added' in response: {data}"

        # Remove paper
        del_resp = requests.delete(
            f"{BACKEND_URL}/topic/{test_topic}/papers/{paper_id}",
            timeout=MEDIUM_TIMEOUT,
        )
        assert del_resp.status_code in [200, 204], (
            f"Remove paper from topic failed: {del_resp.status_code}: {del_resp.text}"
        )


class TestTopicQuery:
    def test_topic_query(self, backend_available, llm_available, test_topic, existing_paper):
        """RAG query against a topic with at least one paper."""
        paper_id = existing_paper.get("paper_id") or existing_paper.get("arxiv_id")

        # Ensure the topic has at least one paper
        requests.post(
            f"{BACKEND_URL}/topic/{test_topic}/papers",
            json={"paper_ids": [paper_id], "similarity_scores": [0.9]},
            timeout=MEDIUM_TIMEOUT,
        )

        resp = requests.post(
            f"{BACKEND_URL}/topic/{test_topic}/query",
            json={"question": "What is the main contribution of these papers?"},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Topic query failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "answer" in data, f"Missing 'answer' in response: {data}"
        assert len(data["answer"]) > 50, (
            f"Topic query answer too short ({len(data['answer'])} chars)"
        )


class TestTopicCreateAndDelete:
    def test_create_verify_delete(self, backend_available, llm_available, cleanup_topic):
        """Create a fresh topic, verify it exists, then delete it.

        Requires llm_available because /topic/create calls the embedding API.
        """
        name = "e2e_temp_topic_to_delete"
        create_resp = requests.post(
            f"{BACKEND_URL}/topic/create",
            json={"name": name, "topic_query": "test"},
            timeout=LLM_TIMEOUT,
        )
        assert create_resp.status_code == 200, (
            f"Topic create failed: {create_resp.status_code}: {create_resp.text}"
        )
        topic_id = create_resp.json().get("topic_id") or create_resp.json().get("id")
        assert topic_id is not None
        cleanup_topic(topic_id)

        # Verify it exists
        check = requests.get(
            f"{BACKEND_URL}/topic/{topic_id}",
            timeout=MEDIUM_TIMEOUT,
        )
        assert check.status_code == 200, f"Topic not found after creation: {check.text}"

        # Delete it
        del_resp = requests.delete(
            f"{BACKEND_URL}/topic/{topic_id}",
            timeout=MEDIUM_TIMEOUT,
        )
        assert del_resp.status_code in [200, 204], (
            f"Topic delete failed: {del_resp.status_code}: {del_resp.text}"
        )

        # Verify it's gone
        gone = requests.get(
            f"{BACKEND_URL}/topic/{topic_id}",
            timeout=MEDIUM_TIMEOUT,
        )
        assert gone.status_code == 404, (
            f"Topic still present after delete: {gone.status_code}"
        )
