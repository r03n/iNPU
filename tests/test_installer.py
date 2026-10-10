"""
Automated validation tests for install.sh installer script.
"""

import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "install.sh"

class TestInstaller(unittest.TestCase):
    def test_installer_exists_and_executable(self):
        self.assertTrue(INSTALL_SH.exists())
        self.assertTrue(os.access(INSTALL_SH, os.X_OK))

    def test_installer_bash_syntax(self):
        res = subprocess.run(["bash", "-n", str(INSTALL_SH)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Syntax errors: {res.stderr}")

    def test_installer_help_flag(self):
        res = subprocess.run([str(INSTALL_SH), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Usage:", res.stdout)
        self.assertIn("--user", res.stdout)
        self.assertIn("--uninstall", res.stdout)
        self.assertIn("--yes", res.stdout)

if __name__ == "__main__":
    unittest.main()
