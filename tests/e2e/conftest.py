"""Shared fixtures for e2e tests.

E2e tests run against the PRODUCTION database (read-only where possible).
Write operations clean up after themselves. Every test gets exactly one
expected status code — no multi-status acceptance lists.

Environment variables:
  BACKEND_URL  — backend address (default: http://localhost:3100)
  SKIP_LLM=1   — skip tests that require a live LLM endpoint
"""
import os
from pathlib import Path
from typing import Optional

import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
TEST_STORAGE = Path(__file__).parent.parent / "storage"
SAMPLE_ARXIV_ID = "1706.03762"

SHORT_TIMEOUT = 10    # health checks, trivial lookups
MEDIUM_TIMEOUT = 60   # metadata, DB queries, embeddings
LLM_TIMEOUT = 300     # single LLM call (summarize, classify, QA)
BATCH_TIMEOUT = 900   # batch operations


# ---------------------------------------------------------------------------
# Session-scoped
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def backend_url():
    return BACKEND_URL


@pytest.fixture(scope="session")
def backend_available(backend_url):
    """Skip entire session if backend is not reachable."""
    try:
        resp = requests.get(f"{backend_url}/health", timeout=SHORT_TIMEOUT)
        if resp.status_code != 200:
            pytest.skip(f"Backend not healthy: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Backend not available at {backend_url}")
    return True


@pytest.fixture(scope="session")
def llm_available():
    """Return True unless SKIP_LLM=1 is set."""
    if os.environ.get("SKIP_LLM", "0") == "1":
        pytest.skip("Skipping LLM test (SKIP_LLM=1)")
    return True


# ---------------------------------------------------------------------------
# Module-scoped
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_arxiv_id():
    """Attention Is All You Need — landmark paper, always useful for tests."""
    return SAMPLE_ARXIV_ID


@pytest.fixture(scope="module")
def sample_pdf_path():
    """Path to the committed sample PDF (1706.03762)."""
    path = TEST_STORAGE / "downloads" / "local" / f"{SAMPLE_ARXIV_ID}.pdf"
    if not path.exists():
        pytest.skip(f"Sample PDF not found at {path}")
    return str(path)


@pytest.fixture(scope="module")
def existing_paper(backend_available, backend_url):
    """Fetch an existing paper from production DB; skip if none exist."""
    resp = requests.get(f"{backend_url}/db/status", timeout=MEDIUM_TIMEOUT)
    assert resp.status_code == 200
    count = resp.json().get("paper_count", 0)
    if count == 0:
        pytest.skip("No papers in production DB — run ingest first")
    # Use well-known paper as the existing paper if it exists
    cached = requests.get(
        f"{backend_url}/papers/{SAMPLE_ARXIV_ID}/cached-data",
        timeout=MEDIUM_TIMEOUT,
    )
    if cached.status_code == 200:
        return {"arxiv_id": SAMPLE_ARXIV_ID, **cached.json()}
    # Fall back to any paper from tree
    tree = requests.get(f"{backend_url}/tree", timeout=MEDIUM_TIMEOUT).json()
    paper_id = _find_first_paper_id(tree)
    if not paper_id:
        pytest.skip("Could not find a paper in the tree")
    return {"arxiv_id": paper_id}


def _find_first_paper_id(node: dict) -> Optional[str]:
    if "paper_id" in node:
        return node["paper_id"]
    for child in node.get("children", []):
        result = _find_first_paper_id(child)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Function-scoped cleanup helpers (yield fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture
def cleanup_paper(backend_url, backend_available):
    """Yield a callable that registers an arxiv_id for deletion on teardown."""
    to_delete = []

    def register(arxiv_id: str):
        to_delete.append(arxiv_id)

    yield register

    for arxiv_id in to_delete:
        try:
            requests.delete(
                f"{backend_url}/papers/{arxiv_id}",
                timeout=MEDIUM_TIMEOUT,
            )
        except Exception:
            pass


@pytest.fixture
def cleanup_topic(backend_url, backend_available):
    """Yield a callable that registers a topic_id for deletion on teardown."""
    to_delete = []

    def register(topic_id: int):
        to_delete.append(topic_id)

    yield register

    for topic_id in to_delete:
        try:
            requests.delete(
                f"{backend_url}/topic/{topic_id}",
                timeout=MEDIUM_TIMEOUT,
            )
        except Exception:
            pass


@pytest.fixture
def cleanup_tree_node(backend_url, backend_available):
    """Yield a callable that registers a node_id for deletion on teardown."""
    to_delete = []

    def register(node_id: str):
        to_delete.append(node_id)

    yield register

    for node_id in to_delete:
        try:
            requests.delete(
                f"{backend_url}/tree/node/{node_id}",
                timeout=MEDIUM_TIMEOUT,
            )
        except Exception:
            pass
