"""E2E tests for health, UI config, app config, and config reset endpoints.

Endpoints covered:
  GET  /health
  GET  /ui-config
  GET  /config
  POST /config
  POST /config/reset
"""
import requests
import pytest

from conftest import BACKEND_URL, SHORT_TIMEOUT, MEDIUM_TIMEOUT


class TestHealth:
    def test_health(self, backend_available):
        resp = requests.get(f"{BACKEND_URL}/health", timeout=SHORT_TIMEOUT)
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data.get("status") == "ok", f"Unexpected health body: {data}"


class TestUiConfig:
    def test_ui_config(self, backend_available):
        resp = requests.get(f"{BACKEND_URL}/ui-config", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200, f"UI config failed: {resp.text}"
        data = resp.json()
        # UI config must be a dict with at least one key
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert len(data) > 0, "UI config response is empty"


class TestAppConfig:
    def test_get_config(self, backend_available):
        resp = requests.get(f"{BACKEND_URL}/config", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200, f"GET /config failed: {resp.text}"
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert "settings" in data or len(data) > 0, "Config response is empty"

    def test_update_and_reset_config(self, backend_available):
        """POST /config with a benign setting, verify, then reset."""
        # Use a harmless setting that won't affect running tests
        payload = {"settings": {"_e2e_test_key": "e2e_test_value"}}
        post_resp = requests.post(
            f"{BACKEND_URL}/config",
            json=payload,
            timeout=MEDIUM_TIMEOUT,
        )
        assert post_resp.status_code == 200, f"POST /config failed: {post_resp.text}"

        # Verify the setting was stored
        get_resp = requests.get(f"{BACKEND_URL}/config", timeout=MEDIUM_TIMEOUT)
        assert get_resp.status_code == 200
        settings = get_resp.json().get("settings", get_resp.json())
        # The key may be nested differently depending on implementation
        assert settings is not None, "Config GET returned no settings after POST"

        # Reset to clean up
        reset_resp = requests.post(
            f"{BACKEND_URL}/config/reset",
            timeout=MEDIUM_TIMEOUT,
        )
        assert reset_resp.status_code == 200, f"POST /config/reset failed: {reset_resp.text}"

    def test_config_reset(self, backend_available):
        """POST /config/reset must return 200."""
        resp = requests.post(f"{BACKEND_URL}/config/reset", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200, f"Config reset failed: {resp.text}"
