"""
Unit and integration tests for iNPU Model Context Protocol (MCP) tools.
"""

import unittest
from intelflow.mcp_server import (
    npu_status,
    ask_npu,
    summarize_code_npu,
    explain_symbol_npu,
    embed_npu,
)

class TestMCP(unittest.TestCase):
    def test_npu_status(self):
        status = npu_status()
        self.assertIsInstance(status, dict)
        self.assertEqual(status["device"], "NPU")
        self.assertTrue(status["npu_available"])
        self.assertIn("hardware_telemetry", status)
        self.assertEqual(status["architecture"], "3720")

    def test_ask_npu(self):
        resp = ask_npu("Say hello in one word", max_tokens=10)
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp.strip()), 0)

    def test_summarize_code_npu(self):
        snippet = "def multiply(x, y): return x * y"
        resp = summarize_code_npu(snippet, language="python")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp.strip()), 0)

    def test_explain_symbol_npu(self):
        resp = explain_symbol_npu("NPUEngine", context="class NPUEngine: pass")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp.strip()), 0)

    def test_embed_npu(self):
        texts = ["Intel Core Ultra NPU", "OpenVINO GenAI runtime"]
        res = embed_npu(texts)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["dimensions"], 384)
        embeddings = res["embeddings"]
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 384)

if __name__ == "__main__":
    unittest.main()
