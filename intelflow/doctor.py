"""
Hardware diagnostics and system verification for Intel NPU.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_kernel_driver() -> dict:
    """Check /dev/accel/accel0 existence and permissions."""
    accel_path = Path("/dev/accel/accel0")
    if not accel_path.exists():
        return {
            "status": False,
            "message": "Missing /dev/accel/accel0. Ensure Linux kernel 6.6+ with intel_vpu module is loaded.",
            "path": None
        }
    
    is_readable = os.access(accel_path, os.R_OK)
    is_writable = os.access(accel_path, os.W_OK)
    
    return {
        "status": is_readable and is_writable,
        "path": str(accel_path),
        "readable": is_readable,
        "writable": is_writable,
        "message": "Node present with full read/write access" if (is_readable and is_writable) else "Permission denied"
    }

def check_openvino_npu() -> dict:
    """Inspect OpenVINO device discovery for NPU."""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        npu_available = "NPU" in devices
        
        info = {
            "status": npu_available,
            "devices": devices,
            "full_name": None,
            "arch": None,
            "caps": None,
            "error": None
        }
        
        if npu_available:
            try:
                info["full_name"] = core.get_property("NPU", "FULL_DEVICE_NAME")
            except Exception:
                info["full_name"] = "Intel NPU"
            try:
                info["arch"] = core.get_property("NPU", "DEVICE_ARCHITECTURE")
            except Exception:
                info["arch"] = "Unknown"
            try:
                info["caps"] = str(core.get_property("NPU", "OPTIMIZATION_CAPABILITIES"))
            except Exception:
                info["caps"] = "Standard"
        else:
            info["error"] = "NPU device not enumerated in OpenVINO. Check Level Zero drivers."
            
        return info
    except Exception as e:
        return {
            "status": False,
            "devices": [],
            "error": str(e)
        }

def run_doctor() -> bool:
    """Run full diagnostic suite and print report."""
    console.print("\n[bold cyan]⚡ iNPU System Diagnostics[/bold cyan]\n")
    
    table = Table(title="Hardware & Driver Readiness", expand=True)
    table.add_column("Component", style="bold white", width=25)
    table.add_column("Status", width=12)
    table.add_column("Details", style="dim")

    # 1. Kernel node
    k_res = check_kernel_driver()
    if k_res["status"]:
        table.add_row("Linux DRM Node", "[green]READY[/green]", f"{k_res['path']} (R/W ok)")
    else:
        table.add_row("Linux DRM Node", "[red]FAILED[/red]", k_res["message"])

    # 2. OpenVINO & NPU discovery
    ov_res = check_openvino_npu()
    if ov_res["status"]:
        table.add_row("OpenVINO Devices", "[green]READY[/green]", f"Available: {', '.join(ov_res['devices'])}")
        table.add_row("Intel NPU Device", "[green]DETECTED[/green]", f"{ov_res['full_name']} (Arch: {ov_res['arch']})")
        table.add_row("NPU Capabilities", "[green]OPTIMAL[/green]", ov_res["caps"])
    else:
        table.add_row("OpenVINO Devices", "[yellow]PARTIAL[/yellow]", f"Available: {', '.join(ov_res.get('devices', []))}")
        table.add_row("Intel NPU Device", "[red]MISSING[/red]", ov_res.get("error", "Not found"))

    console.print(table)

    all_ready = k_res["status"] and ov_res["status"]
    if all_ready:
        console.print(Panel("[bold green]✔ Intel NPU is fully unlocked and ready for inference![/bold green]", border_style="green"))
    else:
        console.print(Panel("[bold yellow]⚠ Some components require attention. Run setup_env.sh to resolve.[/bold yellow]", border_style="yellow"))

    return all_ready

if __name__ == "__main__":
    success = run_doctor()
    sys.exit(0 if success else 1)
