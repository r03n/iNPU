"""
Unit tests for IntelFlowLM doctor and hardware detection.
"""

from intelflow.doctor import check_kernel_driver, check_openvino_npu, run_doctor

def test_kernel_driver_detection():
    res = check_kernel_driver()
    assert isinstance(res, dict)
    assert "status" in res
    assert "path" in res
    # On this Meteor Lake Linux machine, accel0 is present
    assert res["status"] is True
    assert res["path"] == "/dev/accel/accel0"

def test_openvino_npu_detection():
    res = check_openvino_npu()
    assert isinstance(res, dict)
    assert res["status"] is True
    assert "NPU" in res["devices"]
    assert res["full_name"] is not None
    assert "3720" in str(res["arch"]) or res["arch"] is not None

def test_run_doctor():
    success = run_doctor()
    assert success is True
