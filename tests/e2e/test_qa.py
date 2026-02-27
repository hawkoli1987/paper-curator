"""E2E tests for Question & Answer endpoints.

Endpoints covered:
  POST /qa
  POST /qa/structured
"""
import requests
import pytest

from conftest import BACKEND_URL, LLM_TIMEOUT, MEDIUM_TIMEOUT, SAMPLE_ARXIV_ID


class TestQA:
    def test_qa_with_pdf(self, backend_available, llm_available, sample_pdf_path):
        resp = requests.post(
            f"{BACKEND_URL}/qa",
            json={
                "arxiv_id": SAMPLE_ARXIV_ID,
                "question": "What is the main contribution of this paper?",
                "pdf_path": sample_pdf_path,
            },
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"QA failed: {resp.text}"
        data = resp.json()
        assert "answer" in data, f"Missing 'answer' in response: {data}"
        assert len(data["answer"]) > 20, (
            f"Answer too short ({len(data['answer'])} chars)"
        )

    def test_qa_missing_question(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/qa",
            json={"arxiv_id": SAMPLE_ARXIV_ID},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing question, got {resp.status_code}: {resp.text}"
        )


class TestQAStructured:
    def test_qa_structured_with_pdf(self, backend_available, llm_available, sample_pdf_path):
        """POST /qa/structured — StructuredQaRequest requires arxiv_id (no question field)."""
        resp = requests.post(
            f"{BACKEND_URL}/qa/structured",
            json={
                "arxiv_id": SAMPLE_ARXIV_ID,
                "pdf_path": sample_pdf_path,
            },
            timeout=LLM_TIMEOUT,
        )
        assert resp.status_code == 200, f"QA structured failed: {resp.text}"
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"
        assert len(data) > 0, "QA structured response is empty"

    def test_qa_structured_missing_arxiv_id(self, backend_available):
        """StructuredQaRequest requires arxiv_id — empty payload must return 422."""
        resp = requests.post(
            f"{BACKEND_URL}/qa/structured",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )
