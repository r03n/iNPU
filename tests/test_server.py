"""
Contract and regression tests for iNPU dual OpenAI- & Ollama-compatible REST server.
"""

import unittest
from fastapi.testclient import TestClient
from intelflow.server import app

class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["device"], "NPU")

    def test_version_endpoints(self):
        for path in ["/v1/version", "/api/version"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["version"], "0.1.0")
            self.assertEqual(data["arch"], "3720")
            self.assertEqual(data["backend"], "openvino-npu")

    def test_npu_status_endpoint(self):
        response = self.client.get("/api/npu/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("npu_available", data)
        self.assertIn("active_model", data)
        self.assertIn("hardware_telemetry", data)

    def test_list_models_openai(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertGreaterEqual(len(data["data"]), 1)
        model_ids = [m["id"] for m in data["data"]]
        self.assertTrue("qwen2.5" in model_ids or "bge-small" in model_ids)
        for m in data["data"]:
            self.assertEqual(m["owned_by"], "inpu")

    def test_list_models_ollama(self):
        response = self.client.get("/api/tags")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertGreaterEqual(len(data["models"]), 1)

    def test_running_models_ollama(self):
        response = self.client.get("/api/ps")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)

    def test_chat_completions_schema_validation(self):
        # Empty payload fails validation
        response = self.client.post("/v1/chat/completions", json={})
        self.assertEqual(response.status_code, 422)

    def test_text_completions_schema_validation(self):
        # Empty payload fails validation
        response = self.client.post("/v1/completions", json={})
        self.assertEqual(response.status_code, 422)

    def test_ollama_chat_schema_validation(self):
        # Missing messages
        response = self.client.post("/api/chat", json={})
        self.assertEqual(response.status_code, 422)

    def test_ollama_generate_schema_validation(self):
        # Missing prompt
        response = self.client.post("/api/generate", json={})
        self.assertEqual(response.status_code, 422)

    def test_embeddings_schema_validation(self):
        # Missing input
        response = self.client.post("/v1/embeddings", json={})
        self.assertEqual(response.status_code, 422)

    def test_cancel_endpoint(self):
        response = self.client.post("/api/cancel", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_pull_endpoint(self):
        response = self.client.post("/api/pull", json={"model": "qwen2.5"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "pulling")

if __name__ == "__main__":
    unittest.main()
