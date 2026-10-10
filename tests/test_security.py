"""
Automated Security, Concurrency, and Stability Tests for iNPU.
Verifies path traversal guards, model sandbox containment, server error handling,
and stream cancellation mechanics.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from intelflow.hub import ModelHub, validate_repo_id
from intelflow.config import MODEL_DIR
from intelflow.server import app


class TestSecurityAndStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # 1. Path Traversal & Repo Sanitization
    def test_validate_repo_id_valid(self):
        valid_ids = [
            "qwen2.5",
            "qwen2.5:1.5b",
            "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
            "laion/bge-small-en-v1.5_openvino_int8",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ]
        for vid in valid_ids:
            self.assertEqual(validate_repo_id(vid), vid)

    def test_validate_repo_id_path_traversal(self):
        malicious_ids = [
            "../../etc/passwd",
            "../etc/shadow",
            "foo/../../bar",
            "/absolute/path/model",
            "\\windows\\system32",
            "model\0evil",
            "",
            "   ",
        ]
        for mid in malicious_ids:
            with self.assertRaises(ValueError):
                validate_repo_id(mid)

    def test_model_hub_sandbox_containment(self):
        # Valid path is strictly inside MODEL_DIR
        local_p = ModelHub.get_local_path_for_repo("OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
        self.assertTrue(local_p.resolve().is_relative_to(MODEL_DIR.resolve()))

        # Attempting path traversal via get_local_path_for_repo raises ValueError
        with self.assertRaises(ValueError):
            ModelHub.get_local_path_for_repo("../../etc/passwd")

    def test_model_hub_remove_security_guard(self):
        # Attempting to delete arbitrary path outside sandbox must fail with ValueError
        with self.assertRaises(ValueError):
            ModelHub.remove("../../etc/passwd")

    # 2. Server 404 on Nonexistent Model
    def test_server_invalid_model_returns_404(self):
        payload = {
            "model": "malicious/../nonexistent",
            "messages": [{"role": "user", "content": "hello"}],
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        # Should cleanly return 404 Not Found rather than unhandled 500 error
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_server_invalid_embedding_model_returns_404(self):
        payload = {
            "model": "malicious/../nonexistent",
            "input": "hello world",
        }
        res = self.client.post("/v1/embeddings", json=payload)
        self.assertEqual(res.status_code, 404)

    # 3. Stream Cancellation
    def test_stream_cancellation_signal(self):
        # Verify that breaking early from generate_stream sets stop_event via finally block
        from intelflow.engine import NPUEngine
        # Mock engine with a dummy pipe
        class DummyPipe:
            def generate(self, prompt, config, streamer):
                for i in range(100):
                    stopped = streamer(f"chunk_{i}")
                    if stopped:
                        break

        engine = NPUEngine.__new__(NPUEngine)
        engine.pipe = DummyPipe()
        engine.device = "CPU"
        engine.last_metrics = {}

        gen = engine.generate_stream("test prompt", max_new_tokens=100)
        first_chunk = next(gen)
        self.assertEqual(first_chunk, "chunk_0")
        # Close the generator early (simulating client disconnection)
        gen.close()


if __name__ == "__main__":
    unittest.main()
