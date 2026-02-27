"""E2E tests for database administration endpoints.

Endpoints covered:
  GET  /db/status
  POST /db/switch
  POST /db/init

These tests are carefully designed to not corrupt the production database.
The switch/init tests always restore the original database on teardown.
"""
import requests
import pytest

from conftest import BACKEND_URL, MEDIUM_TIMEOUT

TEST_DB_NAME = "paper_curator_e2e_test"


class TestDbStatus:
    def test_db_status(self, backend_available):
        resp = requests.get(f"{BACKEND_URL}/db/status", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200, f"DB status failed: {resp.text}"
        data = resp.json()
        assert "database" in data, f"Missing 'database' in response: {data}"
        assert isinstance(data["database"], str), "database should be a string"
        assert len(data["database"]) > 0, "database name is empty"


class TestDbSwitch:
    def test_db_switch_and_back(self, backend_available):
        """Switch to test DB, verify, switch back — leave production DB intact."""
        # Record current (production) DB
        status_resp = requests.get(f"{BACKEND_URL}/db/status", timeout=MEDIUM_TIMEOUT)
        assert status_resp.status_code == 200
        prod_db = status_resp.json()["database"]

        # Create test DB (idempotent; don't drop existing to preserve any data)
        init_resp = requests.post(
            f"{BACKEND_URL}/db/init",
            json={"database": TEST_DB_NAME, "drop_existing": False},
            timeout=MEDIUM_TIMEOUT,
        )
        assert init_resp.status_code == 200, (
            f"DB init failed: {init_resp.status_code}: {init_resp.text}"
        )

        try:
            # Switch to test DB
            switch_resp = requests.post(
                f"{BACKEND_URL}/db/switch",
                json={"database": TEST_DB_NAME},
                timeout=MEDIUM_TIMEOUT,
            )
            assert switch_resp.status_code == 200, (
                f"DB switch failed: {switch_resp.status_code}: {switch_resp.text}"
            )
            switch_data = switch_resp.json()
            assert "current_database" in switch_data, (
                f"Missing 'current_database' in switch response: {switch_data}"
            )
            assert switch_data["current_database"] == TEST_DB_NAME, (
                f"Expected current_database={TEST_DB_NAME}, got: {switch_data}"
            )

            # Verify via /db/status
            verify_resp = requests.get(f"{BACKEND_URL}/db/status", timeout=MEDIUM_TIMEOUT)
            assert verify_resp.status_code == 200
            assert verify_resp.json()["database"] == TEST_DB_NAME, (
                f"DB status shows wrong DB after switch: {verify_resp.json()}"
            )
        finally:
            # Always switch back to production
            restore_resp = requests.post(
                f"{BACKEND_URL}/db/switch",
                json={"database": prod_db},
                timeout=MEDIUM_TIMEOUT,
            )
            assert restore_resp.status_code == 200, (
                f"CRITICAL: Failed to restore production DB '{prod_db}': "
                f"{restore_resp.status_code}: {restore_resp.text}"
            )

    def test_db_switch_missing_database(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/db/switch",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing database, got {resp.status_code}: {resp.text}"
        )


class TestDbInit:
    def test_db_init_idempotent(self, backend_available):
        """POST /db/init with drop_existing=False is idempotent and safe."""
        resp = requests.post(
            f"{BACKEND_URL}/db/init",
            json={"database": TEST_DB_NAME, "drop_existing": False},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"DB init failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "database" in data, f"Missing 'database' in response: {data}"
        assert "status" in data, f"Missing 'status' in response: {data}"
        assert data["database"] == TEST_DB_NAME, (
            f"Expected database={TEST_DB_NAME}, got: {data['database']}"
        )

    def test_db_init_missing_database(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/db/init",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing database, got {resp.status_code}: {resp.text}"
        )
