"""
IntelFlowLM (inpu): Intel NPU Inference Engine & Developer Runtime
"""

import os
import sys

def _bootstrap_site_packages():
    """Ensure openvino_genai is accessible from system or user venv."""
    try:
        import openvino_genai
        return
    except ImportError:
        pass

    import glob
    candidates = [
        os.environ.get("INTELFLOW_VENV"),
        os.path.expanduser("~/.local/share/intelflow/venv"),
        "/home/wen/Documents/IntelNPU/.venv",
        os.path.join(os.getcwd(), ".venv"),
    ]
    for c in candidates:
        if not c:
            continue
        for sp in glob.glob(os.path.join(c, "lib", "python3.*", "site-packages")):
            if os.path.isdir(sp) and sp not in sys.path:
                sys.path.insert(0, sp)
                try:
                    import openvino_genai
                    return
                except ImportError:
                    pass

def _bootstrap_npu_libs():
    """Ensure Level Zero and OpenVINO NPU libraries are accessible in the dynamic linker path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npu_lib_dir = os.path.abspath(os.path.join(current_dir, "..", "lib", "npu"))

    if os.path.isdir(npu_lib_dir):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if npu_lib_dir not in current_ld.split(":"):
            new_ld = f"{npu_lib_dir}:{current_ld}".strip(":")
            os.environ["LD_LIBRARY_PATH"] = new_ld
            if "INTELFLOW_REEXEC" not in os.environ:
                os.environ["INTELFLOW_REEXEC"] = "1"
                try:
                    args = getattr(sys, "orig_argv", [sys.executable] + sys.argv)
                    os.execve(sys.executable, args, os.environ)
                except Exception:
                    pass

_bootstrap_site_packages()
_bootstrap_npu_libs()

__version__ = "0.1.0"
