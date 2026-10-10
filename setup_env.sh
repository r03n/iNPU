#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "⚡ iNPU: Environment & Driver Setup for Intel NPU"
echo "=========================================================="

# 1. Check Kernel Driver
if [ -e "/dev/accel/accel0" ]; then
    echo "✔ Kernel accelerator node detected: /dev/accel/accel0"
else
    echo "⚠ Warning: /dev/accel/accel0 not found. Ensure linux kernel 6.6+ with intel_vpu module is active."
fi

# 2. Check bundled or system Level Zero libraries
if [ -d "$SCRIPT_DIR/lib/npu" ]; then
    echo "✔ Found bundled NPU Level Zero runtime in lib/npu"
else
    echo "ℹ lib/npu directory not found. System Level Zero drivers in /usr/lib will be used."
fi

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "Installing IntelFlowLM dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -e . -q

echo ""
echo "Running system verification..."
.venv/bin/inpu doctor

echo ""
echo "=========================================================="
echo "✔ Setup completed successfully!"
echo "You can now run:"
echo "  .venv/bin/inpu run qwen2.5       # Interactive NPU chat"
echo "  .venv/bin/inpu serve qwen2.5     # OpenAI REST server (:52625)"
echo "  .venv/bin/inpu --help            # View all options"
echo "=========================================================="
