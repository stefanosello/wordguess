"""Tests for the FastAPI app (server/app.py).

Uses FastAPI's TestClient, which exercises the real ASGI app in-process —
no running server needed.
"""

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health responds 200 with the exact liveness body."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
