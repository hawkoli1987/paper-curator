"""Test health endpoint."""


def test_health(client):
    """Test that the health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
