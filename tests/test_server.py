"""
Contract tests for IntelFlowLM OpenAI-compatible REST server.
"""

from fastapi.testclient import TestClient
from intelflow.server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["device"] == "NPU"

def test_list_models_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    model_ids = [m["id"] for m in data["data"]]
    assert "qwen2.5" in model_ids or "bge-small" in model_ids

def test_chat_completions_schema_validation():
    # Test invalid payload fails validation gracefully
    response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 422  # Validation error
