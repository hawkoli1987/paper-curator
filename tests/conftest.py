import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path

# Add src/backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_arxiv_id():
    """Sample arXiv ID for testing (Attention Is All You Need paper)."""
    return "1706.03762"


@pytest.fixture
def sample_arxiv_url():
    """Sample arXiv URL for testing."""
    return "https://arxiv.org/abs/1706.03762"
