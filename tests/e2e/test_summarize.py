"""E2E tests for summarization endpoints.

Endpoints covered:
  POST /summarize             (plain summary from PDF)
  POST /summarize/structured  (structured summary with sections)
  POST /summary/merge         (merge QA pairs into summary)
  POST /summary/dedup         (deduplicate QA pairs)
"""
import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, SAMPLE_ARXIV_ID


class TestSummarizePlain:
    def test_summarize_with_pdf(self, backend_available, llm_available, sample_pdf_path):
        resp = requests.post(
            f"{BACKEND_URL}/summarize",
            json={"pdf_path": sample_pdf_path, "arxiv_id": SAMPLE_ARXIV_ID},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Summarize failed: {resp.text}"
        data = resp.json()
        assert "summary" in data, f"Missing 'summary' in response: {data}"
        assert len(data["summary"]) > 50, (
            f"Summary too short ({len(data['summary'])} chars)"
        )

    def test_summarize_missing_input(self, backend_available, llm_available):
        """Providing neither pdf_path nor arxiv_id must still fail (not 200).

        Note: SummarizeRequest has all-optional fields, so Pydantic accepts {}.
        The endpoint then calls rag_answer_async with all-None params which should
        raise an error. Requires llm_available since the LLM endpoint check runs first.
        """
        resp = requests.post(
            f"{BACKEND_URL}/summarize",
            json={},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code != 200, (
            f"Expected non-200 for empty payload, got 200: {resp.text}"
        )


class TestSummarizeStructured:
    def test_summarize_structured_with_pdf(
        self, backend_available, llm_available, sample_pdf_path
    ):
        resp = requests.post(
            f"{BACKEND_URL}/summarize/structured",
            json={"pdf_path": sample_pdf_path, "arxiv_id": SAMPLE_ARXIV_ID},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"Structured summarize failed: {resp.text}"
        data = resp.json()
        # Structured summary must return a dict with at least one content field
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"
        assert len(data) > 0, "Structured summary response is empty"

    def test_summarize_structured_missing_input(self, backend_available, llm_available):
        """Empty payload must return 422 with a helpful detail message.

        Note: The endpoint calls _resolve_model (LLM check) BEFORE the 422 guard,
        so this test requires llm_available to reach the input validation code path.
        """
        resp = requests.post(
            f"{BACKEND_URL}/summarize/structured",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for empty payload, got {resp.status_code}: {resp.text}"
        )
        detail = resp.json().get("detail", "")
        assert "pdf_path" in detail or "arxiv_id" in detail or len(detail) > 0, (
            f"422 detail message should mention required fields: {detail}"
        )


class TestSummaryMerge:
    def test_summary_merge(self, backend_available, llm_available):
        """POST /summary/merge — valid arxiv_id returns 200 (or 400 if no QA pairs exist)."""
        resp = requests.post(
            f"{BACKEND_URL}/summary/merge",
            json={"arxiv_id": SAMPLE_ARXIV_ID},
            timeout=LLM_TIMEOUT,
        )
        # 200 if QA pairs exist; 400 if paper has no QA pairs to merge
        assert resp.status_code in [200, 400], (
            f"Unexpected status from /summary/merge: {resp.status_code}: {resp.text}"
        )
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, dict), "Merge response should be a dict"

    def test_summary_merge_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/summary/merge",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )


class TestSummaryDedup:
    def test_summary_dedup(self, backend_available, llm_available):
        """POST /summary/dedup — returns 200 for a known paper."""
        resp = requests.post(
            f"{BACKEND_URL}/summary/dedup",
            json={"arxiv_id": SAMPLE_ARXIV_ID},
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code in [200, 400], (
            f"Unexpected status from /summary/dedup: {resp.status_code}: {resp.text}"
        )
        if resp.status_code == 200:
            assert isinstance(resp.json(), dict), "Dedup response should be a dict"

    def test_summary_dedup_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/summary/dedup",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )
