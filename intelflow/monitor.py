"""
Hardware telemetry and live monitor for Intel NPU (Meteor Lake / Core Ultra).
Reads kernel sysfs counters (/sys/class/accel/accel0/device/) and active device handles.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

SYSFS_ACCEL_DIR = Path("/sys/class/accel/accel0/device")
DEV_ACCEL_NODE = Path("/dev/accel/accel0")


class NPUTelemetry:
    """Probes Linux DRM accel sysfs telemetry and active process bindings."""

    def __init__(self, sysfs_dir: Path = SYSFS_ACCEL_DIR):
        self.sysfs_dir = sysfs_dir
        self.last_busy_time_us: Optional[int] = None
        self.last_sample_time: Optional[float] = None

    def is_npu_available(self) -> bool:
        return self.sysfs_dir.exists()

    def _read_int(self, filename: str, default: int = 0) -> int:
        p = self.sysfs_dir / filename
        if p.exists():
            try:
                return int(p.read_text().strip())
            except Exception:
                return default
        return default

    def _read_str(self, filename: str, default: str = "Unknown") -> str:
        p = self.sysfs_dir / filename
        if p.exists():
            try:
                return p.read_text().strip()
            except Exception:
                return default
        return default

    def sample(self) -> Dict[str, Any]:
        """Collect instantaneous hardware metrics and compute interval utilization."""
        now = time.perf_counter()
        busy_us = self._read_int("npu_busy_time_us", 0)
        cur_freq = self._read_int("npu_current_frequency_mhz", 0)
        max_freq = self._read_int("npu_max_frequency_mhz", 1400)
        mem_bytes = self._read_int("npu_memory_utilization", 0)
        pwr_state = self._read_str("power_state", "Unknown")

        utilization_pct = 0.0
        if self.last_busy_time_us is not None and self.last_sample_time is not None:
            delta_busy_us = max(0, busy_us - self.last_busy_time_us)
            delta_time_us = max(1, (now - self.last_sample_time) * 1_000_000)
            utilization_pct = min(100.0, max(0.0, (delta_busy_us / delta_time_us) * 100.0))

        self.last_busy_time_us = busy_us
        self.last_sample_time = now

        return {
            "available": self.is_npu_available(),
            "utilization_pct": round(utilization_pct, 1),
            "current_freq_mhz": cur_freq,
            "max_freq_mhz": max_freq,
            "power_state": pwr_state,
            "memory_mb": round(mem_bytes / (1024 * 1024), 2),
            "memory_bytes": mem_bytes,
            "busy_time_us": busy_us,
            "busy_time_sec": round(busy_us / 1_000_000, 2),
        }

    @staticmethod
    def get_active_processes() -> List[Dict[str, Any]]:
        """Identify processes actively holding open file descriptors to /dev/accel/accel0."""
        target_dev = DEV_ACCEL_NODE.resolve() if DEV_ACCEL_NODE.exists() else None
        if not target_dev:
            return []

        procs: List[Dict[str, Any]] = []
        proc_dir = Path("/proc")
        for p in proc_dir.iterdir():
            if not (p.is_dir() and p.name.isdigit()):
                continue
            fd_dir = p / "fd"
            try:
                for fd in fd_dir.iterdir():
                    try:
                        if fd.resolve() == target_dev:
                            pid = p.name
                            cmdline = ""
                            cmdline_path = p / "cmdline"
                            if cmdline_path.exists():
                                cmdline = cmdline_path.read_text().replace("\x00", " ").strip()
                            comm_path = p / "comm"
                            comm = comm_path.read_text().strip() if comm_path.exists() else "unknown"
                            procs.append({
                                "pid": pid,
                                "name": comm,
                                "cmdline": cmdline or comm,
                                "device": str(target_dev),
                            })
                            break
                    except (PermissionError, FileNotFoundError):
                        pass
            except (PermissionError, FileNotFoundError):
                pass
        return procs


def render_dashboard(metrics: Dict[str, Any], processes: List[Dict[str, Any]]) -> Panel:
    """Render a terminal dashboard for NPU activity."""
    # Top hardware metrics table
    hw_table = Table(expand=True, box=None, show_header=False)
    hw_table.add_column("Key", style="bold cyan", width=22)
    hw_table.add_column("Value", style="bold white")
    hw_table.add_column("Details", style="dim")

    # Power status styling
    pwr = metrics.get("power_state", "Unknown")
    if pwr == "D0":
        pwr_text = "[bold green]D0 (Active Compute)[/bold green]"
    elif "D3" in pwr:
        pwr_text = "[dim yellow]D3hot (Power-Gated / Idle)[/dim yellow]"
    else:
        pwr_text = f"[white]{pwr}[/white]"

    util = metrics.get("utilization_pct", 0.0)
    util_color = "green" if util > 0 else "dim"
    freq = metrics.get("current_freq_mhz", 0)
    max_freq = metrics.get("max_freq_mhz", 1400)

    hw_table.add_row("Device", "Intel(R) AI Boost (NPU 3720)", "Meteor Lake / Core Ultra")
    hw_table.add_row("Compute Utilization", f"[{util_color}]{util:.1f}%[/{util_color}]", "Silicon active duty cycle")
    hw_table.add_row("Current Clock", f"{freq} MHz / {max_freq} MHz", "Dynamic frequency scaling")
    hw_table.add_row("Power State", pwr_text, "Ultra-low power (~2-4W)")
    hw_table.add_row("Memory Allocated", f"{metrics.get('memory_mb', 0.0)} MB", f"{metrics.get('memory_bytes', 0):,} bytes")
    hw_table.add_row("Cumulative Compute", f"{metrics.get('busy_time_sec', 0.0)} seconds", f"{metrics.get('busy_time_us', 0):,} µs hardware counter")

    # Process table
    proc_table = Table(title="Active Processes on /dev/accel/accel0", expand=True, show_edge=False)
    proc_table.add_column("PID", style="bold green", width=10)
    proc_table.add_column("Process", style="bold white", width=18)
    proc_table.add_column("Command Line", style="dim", overflow="ellipsis")

    if processes:
        for p in processes:
            proc_table.add_row(p["pid"], p["name"], p["cmdline"][:70])
    else:
        proc_table.add_row("-", "[dim]No active clients[/dim]", "[dim]NPU idle, awaiting Level Zero inference stream[/dim]")

    content = Table.grid(expand=True)
    content.add_row(hw_table)
    content.add_row(Text(""))
    content.add_row(proc_table)

    return Panel(
        content,
        title="⚡ [bold cyan]iNPU Hardware Monitor[/bold cyan]",
        subtitle="[dim]Press Ctrl+C to exit[/dim]",
        border_style="cyan",
        expand=True,
    )


def run_monitor(interval: float = 1.0, once: bool = False):
    """Run interactive or one-shot NPU monitor."""
    console = Console()
    telemetry = NPUTelemetry()

    if not telemetry.is_npu_available():
        console.print("[bold red]❌ Intel NPU sysfs node not found at /sys/class/accel/accel0/device/[/bold red]")
        console.print("[dim]Ensure intel_vpu kernel module is loaded and /dev/accel/accel0 exists.[/dim]")
        sys.exit(1)

    # Initial sample
    telemetry.sample()
    if once:
        time.sleep(0.1)
        metrics = telemetry.sample()
        procs = telemetry.get_active_processes()
        console.print(render_dashboard(metrics, procs))
        return

    console.print("[dim]Starting real-time Intel NPU monitor...[/dim]")
    try:
        with Live(console=console, screen=False, refresh_per_second=int(max(1, 1 / interval))) as live:
            while True:
                time.sleep(interval)
                metrics = telemetry.sample()
                procs = telemetry.get_active_processes()
                live.update(render_dashboard(metrics, procs))
    except KeyboardInterrupt:
        pass
