"""
Automated CLI integration tests for iNPU.
"""

import json
import unittest
from typer.testing import CliRunner
from intelflow.cli import app

class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("iNPU", result.stdout)
        self.assertIn("doctor", result.stdout)
        self.assertIn("validate", result.stdout)
        self.assertIn("run", result.stdout)
        self.assertIn("serve", result.stdout)
        self.assertIn("benchmark", result.stdout)
        self.assertIn("top", result.stdout)
        self.assertIn("mcp", result.stdout)

    def test_cli_version(self):
        res1 = self.runner.invoke(app, ["--version"])
        self.assertEqual(res1.exit_code, 0)
        self.assertIn("iNPU version 0.1.0", res1.stdout)

        res2 = self.runner.invoke(app, ["-v"])
        self.assertEqual(res2.exit_code, 0)
        self.assertIn("iNPU version 0.1.0", res2.stdout)

    def test_cli_port(self):
        result = self.runner.invoke(app, ["port"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("52625", result.stdout)

    def test_cli_validate_json(self):
        result = self.runner.invoke(app, ["validate", "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.stdout)
        self.assertTrue(data["valid"])
        self.assertEqual(data["drm_status"], "READY")
        self.assertEqual(data["openvino_status"], "OPTIMAL")

    def test_cli_list_json(self):
        result = self.runner.invoke(app, ["list", "--json"])
        self.assertEqual(result.exit_code, 0)
        models = json.loads(result.stdout)
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 5)
        tags = [m["tag"] for m in models]
        self.assertIn("qwen2.5", tags)
        self.assertIn("bge-small", tags)

    def test_cli_list_quiet(self):
        result = self.runner.invoke(app, ["list", "-q"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("qwen2.5", result.stdout)

    def test_cli_list_filter_installed(self):
        result = self.runner.invoke(app, ["list", "--json", "--filter", "installed"])
        self.assertEqual(result.exit_code, 0)
        models = json.loads(result.stdout)
        for m in models:
            self.assertTrue(m["installed"])

    def test_cli_list_filter_not_installed(self):
        result = self.runner.invoke(app, ["list", "--json", "--filter", "not-installed"])
        self.assertEqual(result.exit_code, 0)
        models = json.loads(result.stdout)
        for m in models:
            self.assertFalse(m["installed"])

    def test_cli_list_filter_invalid(self):
        result = self.runner.invoke(app, ["list", "--filter", "invalid_filter"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Invalid filter", result.stdout)

    def test_cli_check_model_json(self):
        result = self.runner.invoke(app, ["check", "qwen2.5", "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.stdout)
        self.assertTrue(data["installed"])
        self.assertTrue(data["valid"])
        self.assertTrue(data["has_model_ir"])
        self.assertTrue(data["has_weights"])

    def test_cli_top_once(self):
        result = self.runner.invoke(app, ["top", "--once"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("iNPU Hardware Monitor", result.stdout)
        self.assertIn("Intel(R) AI Boost", result.stdout)

if __name__ == "__main__":
    unittest.main()
