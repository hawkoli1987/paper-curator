"""E2E tests for paper CRUD operations.

Endpoints covered:
  POST /papers/save
  GET  /papers/{arxiv_id}/cached-data
  DELETE /papers/{arxiv_id}
  POST /papers/prefetch
"""
import requests
import pytest

from conftest import BACKEND_URL, MEDIUM_TIMEOUT, LLM_TIMEOUT, SAMPLE_ARXIV_ID


# A distinct arXiv ID used for save/delete tests (BERT paper)
CRUD_ARXIV_ID = "1810.04805"


class TestPaperSave:
    def test_save_paper_with_pdf(
        self, backend_available, llm_available, sample_pdf_path, cleanup_paper
    ):
        """Save a new paper and verify it is indexed (embedding computed)."""
        # Resolve metadata first
        resolve_resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": SAMPLE_ARXIV_ID},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resolve_resp.status_code == 200
        meta = resolve_resp.json()

        save_resp = requests.post(
            f"{BACKEND_URL}/papers/save",
            json={
                "arxiv_id": SAMPLE_ARXIV_ID,
                "title": meta["title"],
                "authors": meta.get("authors", []),
                "abstract": meta.get("summary", ""),
                "pdf_path": sample_pdf_path,
            },
            timeout=LLM_TIMEOUT,
        )
        # 200 = newly saved, 409 = already exists
        assert save_resp.status_code in [200, 409], (
            f"Save paper failed: {save_resp.status_code}: {save_resp.text}"
        )
        if save_resp.status_code == 200:
            data = save_resp.json()
            assert data.get("indexed") is True, (
                f"Paper saved but indexed={data.get('indexed')} — embedding not computed"
            )
            # Register for cleanup in case the paper wasn't in the DB before
            cleanup_paper(SAMPLE_ARXIV_ID)

    def test_save_paper_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/save",
            json={"title": "Some Title"},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )


class TestPaperCachedData:
    def test_cached_data_existing_paper(self, backend_available, existing_paper):
        arxiv_id = existing_paper["arxiv_id"]
        resp = requests.get(
            f"{BACKEND_URL}/papers/{arxiv_id}/cached-data",
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"GET /papers/{arxiv_id}/cached-data failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"

    def test_cached_data_nonexistent_paper(self, backend_available):
        resp = requests.get(
            f"{BACKEND_URL}/papers/xxxx.99999/cached-data",
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 404, (
            f"Expected 404 for nonexistent paper, got {resp.status_code}: {resp.text}"
        )


class TestPaperDelete:
    def test_save_then_delete(self, backend_available, llm_available, sample_pdf_path):
        """Save a paper, verify it is in DB, then delete it, verify it is gone."""
        # Use BERT paper to avoid interfering with the sample paper used elsewhere
        resolve_resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": CRUD_ARXIV_ID},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resolve_resp.status_code == 200, f"Resolve failed: {resolve_resp.text}"
        meta = resolve_resp.json()

        # Download PDF to scratch
        import os, shutil
        from pathlib import Path
        scratch = str(Path(__file__).parent.parent / "storage" / "downloads" / "crud_scratch")
        os.makedirs(scratch, exist_ok=True)
        try:
            dl_resp = requests.post(
                f"{BACKEND_URL}/arxiv/download",
                json={"arxiv_id": CRUD_ARXIV_ID, "output_dir": scratch},
                timeout=MEDIUM_TIMEOUT,
            )
            assert dl_resp.status_code == 200, f"Download failed: {dl_resp.text}"
            pdf_path = dl_resp.json()["pdf_path"]

            save_resp = requests.post(
                f"{BACKEND_URL}/papers/save",
                json={
                    "arxiv_id": CRUD_ARXIV_ID,
                    "title": meta["title"],
                    "authors": meta.get("authors", []),
                    "abstract": meta.get("summary", ""),
                    "pdf_path": pdf_path,
                },
                timeout=LLM_TIMEOUT,
            )
            assert save_resp.status_code in [200, 409], (
                f"Save failed: {save_resp.status_code}: {save_resp.text}"
            )

            # Delete the paper
            del_resp = requests.delete(
                f"{BACKEND_URL}/papers/{CRUD_ARXIV_ID}",
                timeout=MEDIUM_TIMEOUT,
            )
            assert del_resp.status_code in [200, 204], (
                f"Delete failed: {del_resp.status_code}: {del_resp.text}"
            )

            # Verify it's gone
            check_resp = requests.get(
                f"{BACKEND_URL}/papers/{CRUD_ARXIV_ID}/cached-data",
                timeout=MEDIUM_TIMEOUT,
            )
            assert check_resp.status_code == 404, (
                f"Paper still present after delete: {check_resp.status_code}"
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_delete_nonexistent_paper(self, backend_available):
        resp = requests.delete(
            f"{BACKEND_URL}/papers/xxxx.99999",
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code in [404, 204], (
            f"Expected 404/204 for nonexistent paper delete, got {resp.status_code}: {resp.text}"
        )


class TestPaperPrefetch:
    def test_prefetch_paper(self, backend_available, llm_available):
        """POST /papers/prefetch — pre-download and index a paper."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/prefetch",
            json={"arxiv_id": SAMPLE_ARXIV_ID},
            timeout=LLM_TIMEOUT,
        )
        # 200 = prefetched, 409 = already exists
        assert resp.status_code in [200, 409], (
            f"Prefetch failed: {resp.status_code}: {resp.text}"
        )
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, dict), f"Expected dict response, got {type(data)}"

    def test_prefetch_missing_arxiv_id(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/prefetch",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing arxiv_id, got {resp.status_code}: {resp.text}"
        )
