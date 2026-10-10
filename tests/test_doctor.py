"""
Unit tests for iNPU doctor and hardware detection.
"""

import unittest
from intelflow.doctor import check_kernel_driver, check_openvino_npu, run_doctor

class TestDoctor(unittest.TestCase):
    def test_kernel_driver_detection(self):
        res = check_kernel_driver()
        self.assertIsInstance(res, dict)
        self.assertIn("status", res)
        self.assertIn("path", res)
        # On this Meteor Lake Linux machine, accel0 is present
        self.assertTrue(res["status"])
        self.assertEqual(res["path"], "/dev/accel/accel0")

    def test_openvino_npu_detection(self):
        res = check_openvino_npu()
        self.assertIsInstance(res, dict)
        self.assertTrue(res["status"])
        self.assertIn("NPU", res["devices"])
        self.assertIsNotNone(res["full_name"])
        self.assertTrue("3720" in str(res["arch"]) or res["arch"] is not None)

    def test_run_doctor(self):
        success = run_doctor()
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
