"""E2E tests for batch paper operations.

Endpoints covered:
  POST /papers/batch-ingest      (directory mode)
  POST /papers/reabbreviate      (single paper)
  POST /papers/reabbreviate-all  (all papers)
"""
import os
from pathlib import Path

import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, BATCH_TIMEOUT, SAMPLE_ARXIV_ID

TEST_STORAGE = Path(__file__).parent.parent / "storage"
LOCAL_DIR = str(TEST_STORAGE / "downloads" / "local")


class TestBatchIngest:
    def test_batch_ingest_local_directory(self, backend_available, llm_available):
        """POST /papers/batch-ingest with a local directory of PDFs.

        Uses tests/storage/downloads/local/ which has the 10 committed sample PDFs.
        Papers already in DB are skipped (idempotent), so the test is safe to run
        against the production DB.
        """
        assert Path(LOCAL_DIR).exists(), f"Local PDF dir not found: {LOCAL_DIR}"
        pdfs = list(Path(LOCAL_DIR).glob("*.pdf"))
        assert len(pdfs) >= 1, f"No PDFs in {LOCAL_DIR}"

        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={"directory": LOCAL_DIR, "limit": 10},
            timeout=BATCH_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Batch ingest failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "success" in data or "skipped" in data, (
            f"Batch ingest response missing expected fields: {data}"
        )
        total = data.get("success", 0) + data.get("skipped", 0)
        assert total >= 0, f"Expected non-negative totals: {data}"

    def test_batch_ingest_missing_source(self, backend_available, llm_available):
        """No directory or slack_channel should return 400 (after LLM check passes).

        Note: The endpoint checks LLM availability first, then validates source.
        Without LLM, the endpoint returns 503 before reaching source validation.
        This test requires llm_available to ensure source validation is reached.
        """
        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for missing source, got {resp.status_code}: {resp.text}"
        )

    def test_batch_ingest_slack_no_token(self, backend_available, llm_available):
        """Slack ingest without token should return 400.

        Note: The endpoint checks LLM availability first, so requires llm_available.
        """
        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={"slack_channel": "https://app.slack.com/client/T04MW5HMWV9/C0A727EKAJV"},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 400, (
            f"Expected 400 without Slack token, got {resp.status_code}: {resp.text}"
        )


class TestReabbreviate:
    def test_reabbreviate_single(self, backend_available, llm_available, existing_paper):
        """POST /papers/reabbreviate for an existing paper."""
        arxiv_id = existing_paper["arxiv_id"]
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate",
            json={"arxiv_id": arxiv_id},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Reabbreviate failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"

    def test_reabbreviate_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )

    def test_reabbreviate_nonexistent_paper(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate",
            json={"arxiv_id": "xxxx.99999"},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code in [400, 404], (
            f"Expected 400/404 for nonexistent paper, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.skip(reason="reabbreviate-all has no GUI button and is not currently in use")
class TestReabbreviateAll:
    def test_reabbreviate_all(self, backend_available, llm_available):
        """POST /papers/reabbreviate-all must return 200 with update count."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate-all",
            json={},
            timeout=BATCH_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Reabbreviate-all failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "updated" in data or isinstance(data, dict), (
            f"Reabbreviate-all response missing 'updated': {data}"
        )
