"""Connectivity tests - verify all required services are accessible."""
import os
import requests
import psycopg2

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")


def get_setting_value(settings: dict, key: str):
    """Extract value from settings, handling both flat and nested formats."""
    val = settings.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("value")
    return val


class TestConnectivity:
    """Verify all services are accessible."""

    def test_backend_health(self):
        """Backend API is accessible and healthy."""
        resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_llm_endpoint_accessible(self):
        """LLM endpoint is accessible via backend config."""
        resp = requests.get(f"{BACKEND_URL}/config", timeout=10)
        assert resp.status_code == 200
        config = resp.json()

        settings = config.get("settings", {})
        llm_base_url = get_setting_value(settings, "llm_base_url")
        assert llm_base_url, "LLM base URL not configured"
        assert llm_base_url.startswith(("http://", "https://")), f"LLM base URL invalid: {llm_base_url}"

        # Include ngrok bypass header — required for ngrok free-tier URLs, harmless otherwise
        headers = {"ngrok-skip-browser-warning": "true"}
        models_resp = requests.get(f"{llm_base_url}/models", timeout=15, headers=headers)
        assert models_resp.status_code == 200, f"LLM endpoint not responding at {llm_base_url}"

        models = models_resp.json()
        assert "data" in models, "LLM models response missing 'data'"
        assert len(models["data"]) > 0, "No LLM models available"

    def test_embedding_endpoint_accessible(self):
        """Embedding endpoint is accessible via backend config."""
        resp = requests.get(f"{BACKEND_URL}/config", timeout=10)
        assert resp.status_code == 200
        config = resp.json()

        settings = config.get("settings", {})
        embed_base_url = get_setting_value(settings, "embedding_base_url") or get_setting_value(settings, "embed_base_url")
        assert embed_base_url, "Embedding base URL not configured"
        assert embed_base_url.startswith(("http://", "https://")), f"Embedding base URL invalid: {embed_base_url}"

        # Include ngrok bypass header — required for ngrok free-tier URLs, harmless otherwise
        headers = {"ngrok-skip-browser-warning": "true"}
        models_resp = requests.get(f"{embed_base_url}/models", timeout=15, headers=headers)
        assert models_resp.status_code == 200, f"Embedding endpoint not responding at {embed_base_url}"

        models = models_resp.json()
        assert "data" in models, "Embedding models response missing 'data'"
        assert len(models["data"]) > 0, "No embedding models available"

    def test_frontend_accessible(self):
        """Frontend is accessible."""
        frontend_url = "http://localhost:3000"
        
        frontend_resp = requests.get(frontend_url, timeout=10)
        assert frontend_resp.status_code in [200, 301, 302, 304], \
            f"Frontend not accessible at {frontend_url}: status {frontend_resp.status_code}"

    def test_database_accessible(self):
        """PostgreSQL database is accessible."""
        conn = psycopg2.connect(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            database=os.environ.get("PGDATABASE", "paper_curator"),
            user=os.environ.get("PGUSER", "curator"),
            password=os.environ.get("PGPASSWORD", "curator123"),
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            assert result[0] == 1
            
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ext = cur.fetchone()
            assert ext is not None, "pgvector extension not installed"
        conn.close()

    def test_config_obtainable(self):
        """All configuration values are obtainable."""
        resp = requests.get(f"{BACKEND_URL}/config", timeout=10)
        assert resp.status_code == 200
        config = resp.json()
        
        assert "settings" in config, "Config missing 'settings' key"
        settings = config["settings"]
        
        assert len(settings) > 0, "Settings is empty"

    def test_ui_config_obtainable(self):
        """UI-specific config is obtainable."""
        resp = requests.get(f"{BACKEND_URL}/ui-config", timeout=10)
        assert resp.status_code == 200
        config = resp.json()
        
        expected_keys = ["hover_debounce_ms", "max_similar_papers", "tree_auto_save_interval_ms"]
        for key in expected_keys:
            assert key in config, f"UI config missing '{key}'"
            assert isinstance(config[key], (int, float)), f"{key} should be numeric"
            assert config[key] > 0, f"{key} should be positive"
