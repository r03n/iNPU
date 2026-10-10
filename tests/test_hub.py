"""
Unit tests for iNPU Model Hub and registry.
"""

import unittest
from intelflow.hub import ModelHub
from intelflow.config import MODEL_REGISTRY, resolve_model_tag

class TestHub(unittest.TestCase):
    def test_resolve_model_tag_registry(self):
        self.assertEqual(resolve_model_tag("qwen2.5"), "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
        self.assertEqual(resolve_model_tag("bge-small"), "laion/bge-small-en-v1.5_openvino_int8")
        self.assertEqual(resolve_model_tag("bge-base"), "OpenVINO/bge-base-en-v1.5-int8-ov")

    def test_resolve_model_tag_direct(self):
        direct_repo = "CustomOrg/CustomModel-ov"
        self.assertEqual(resolve_model_tag(direct_repo), direct_repo)

    def test_list_catalog(self):
        catalog = ModelHub.list_catalog()
        self.assertIsInstance(catalog, dict)
        self.assertIn("qwen2.5", catalog)
        self.assertIn("llama3.2:1b", catalog)
        self.assertIn("bge-small", catalog)

    def test_list_installed(self):
        installed = ModelHub.list_installed()
        self.assertIsInstance(installed, list)
        if installed:
            first = installed[0]
            self.assertIn("tag", first)
            self.assertIn("repo_id", first)
            self.assertIn("path", first)
            self.assertIn("size_mb", first)

    def test_check_installed_model(self):
        res = ModelHub.check("qwen2.5")
        self.assertIsInstance(res, dict)
        self.assertTrue(res["installed"])
        self.assertTrue(res["valid"])
        self.assertGreater(res["size_mb"], 100)

    def test_check_nonexistent_model(self):
        res = ModelHub.check("nonexistent-model-xyz")
        self.assertIsInstance(res, dict)
        self.assertFalse(res["installed"])
        self.assertFalse(res["valid"])

if __name__ == "__main__":
    unittest.main()
