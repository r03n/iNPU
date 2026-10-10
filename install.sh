#!/usr/bin/env bash
# ==============================================================================
# iNPU — Universal Linux Installer
# High-performance NPU-first LLM runtime for Intel Core Ultra processors.
# ==============================================================================

set -eo pipefail

APP_NAME="iNPU"
CMD_NAME="inpu"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_INSTALL_DIR="/opt/inpu"
SYSTEM_BIN_DIR="/usr/local/bin"
USER_INSTALL_DIR="$HOME/.local/share/inpu"
USER_BIN_DIR="$HOME/.local/bin"

INSTALL_MODE="system"
NON_INTERACTIVE=false

# ------------------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}==>${NC} ${BOLD}$1${NC}"; }
success() { echo -e "${GREEN}✔${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✖${NC} $1" >&2; }

# ------------------------------------------------------------------------------
# CLI Arguments
# ------------------------------------------------------------------------------
show_help() {
    cat << EOF
iNPU Installer — Ultra-efficient Intel NPU runtime

Usage:
  ./install.sh [options]

Options:
  --user          Install only in current user home (~/.local/bin and ~/.local/share/inpu)
  --yes, -y       Non-interactive mode (automatically proceed with defaults)
  --uninstall     Remove installed binaries, virtual environment, and udev rules
  --help, -h      Show this help message

Examples:
  ./install.sh           # System-wide installation (requires sudo)
  ./install.sh --user    # User-space installation (no root required)
  ./install.sh --yes     # Automated script/CI installation
EOF
    exit 0
}

do_uninstall() {
    info "Uninstalling ${APP_NAME}..."
    if [ -f "$SYSTEM_BIN_DIR/$CMD_NAME" ]; then
        sudo rm -f "$SYSTEM_BIN_DIR/$CMD_NAME"
        success "Removed $SYSTEM_BIN_DIR/$CMD_NAME"
    fi
    if [ -f "$USER_BIN_DIR/$CMD_NAME" ]; then
        rm -f "$USER_BIN_DIR/$CMD_NAME"
        success "Removed $USER_BIN_DIR/$CMD_NAME"
    fi
    if [ -d "$SYSTEM_INSTALL_DIR" ]; then
        sudo rm -rf "$SYSTEM_INSTALL_DIR"
        success "Removed $SYSTEM_INSTALL_DIR"
    fi
    if [ -d "$USER_INSTALL_DIR" ]; then
        rm -rf "$USER_INSTALL_DIR"
        success "Removed $USER_INSTALL_DIR"
    fi
    if [ -f "/etc/udev/rules.d/99-inpu-accel.rules" ]; then
        sudo rm -f "/etc/udev/rules.d/99-inpu-accel.rules"
        sudo udevadm control --reload && sudo udevadm trigger || true
        success "Removed udev rules"
    fi
    success "${APP_NAME} uninstalled successfully."
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            INSTALL_MODE="user"
            shift
            ;;
        --yes|-y)
            NON_INTERACTIVE=true
            shift
            ;;
        --uninstall)
            do_uninstall
            ;;
        --help|-h)
            show_help
            ;;
        *)
            error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------------------
cat << "EOF"
  _ _   _ _____  _   _ 
 (_) \ | |  __ \| | | |
  _|  \| | |__) | | | |
 | | . ` |  ___/| | | |
 | | |\  | |    | |_| |
 |_|_| \_|_|     \___/ 
 ⚡ Ultra-Efficient Intel Core Ultra NPU Runtime
EOF
echo ""

# ------------------------------------------------------------------------------
# 1. System & Hardware Detection
# ------------------------------------------------------------------------------
info "Probing hardware and operating system..."

KERNEL_VER=$(uname -r)
info "Linux Kernel: $KERNEL_VER"

# Check Linux distribution
DISTRO="generic"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO="$ID"
    info "Detected Distribution: $NAME"
fi

# Check for Intel NPU accelerator node
ACCEL_NODE="/dev/accel/accel0"
if [ -e "$ACCEL_NODE" ]; then
    success "Intel NPU accelerator node detected at $ACCEL_NODE"
else
    warn "NPU node $ACCEL_NODE not found. Ensure Linux kernel 6.6+ and intel_vpu module are loaded."
    warn "Inference will run with fallback acceleration until kernel driver is active."
fi

# Check Python version
PYTHON_BIN=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON_BIN" ]; then
    error "Python 3 is required but not installed."
    exit 1
fi
PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Detected Python: $PY_VER ($PYTHON_BIN)"

# ------------------------------------------------------------------------------
# 2. Package Manager Dependencies
# ------------------------------------------------------------------------------
install_system_deps() {
    case "$DISTRO" in
        arch|manjaro|endeavouros)
            info "Arch-based system detected. Checking system dependencies..."
            if command -v pacman &>/dev/null; then
                local pkgs=("intel-npu-driver" "intel-npu-compiler" "level-zero-loader")
                local missing=()
                for p in "${pkgs[@]}"; do
                    if ! pacman -Q "$p" &>/dev/null; then
                        missing+=("$p")
                    fi
                done
                if [ ${#missing[@]} -gt 0 ]; then
                    info "Installing driver dependencies: ${missing[*]}"
                    sudo pacman -S --needed --noconfirm "${missing[@]}" || true
                fi
            fi
            ;;
        ubuntu|debian|pop)
            info "Debian/Ubuntu system detected. Ensuring prerequisites..."
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq || true
                sudo apt-get install -y -qq python3-venv python3-pip libze-loader || true
            fi
            ;;
        fedora|rhel)
            info "Fedora/RHEL system detected. Ensuring prerequisites..."
            if command -v dnf &>/dev/null; then
                sudo dnf install -y python3-devel level-zero || true
            fi
            ;;
        *)
            info "Proceeding with generic Linux installation..."
            ;;
    esac
}

if [ "$INSTALL_MODE" = "system" ]; then
    install_system_deps
fi

# ------------------------------------------------------------------------------
# 3. Udev Rules and User Permissions
# ------------------------------------------------------------------------------
setup_permissions() {
    info "Configuring hardware permissions for /dev/accel/accel*..."
    local UDEV_RULE_FILE="/etc/udev/rules.d/99-inpu-accel.rules"
    local RULE_CONTENT='SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0666", TAG+="uaccess"'

    if [ "$INSTALL_MODE" = "system" ] || sudo -v 2>/dev/null; then
        echo "$RULE_CONTENT" | sudo tee "$UDEV_RULE_FILE" >/dev/null
        sudo udevadm control --reload 2>/dev/null || true
        sudo udevadm trigger 2>/dev/null || true
        success "Installed udev rule to $UDEV_RULE_FILE (MODE=0666, TAG+=uaccess)"

        # Ensure user is in group render
        if id -nG "$USER" | grep -qw "render"; then
            success "User '$USER' is already in group 'render'"
        else
            sudo usermod -aG render "$USER" 2>/dev/null || true
            success "Added user '$USER' to group 'render'"
        fi
    else
        warn "Skipping root udev setup in user mode. Run with sudo if permission denied on /dev/accel/accel0."
    fi
}

setup_permissions

# ------------------------------------------------------------------------------
# 4. Isolated Virtual Environment Setup
# ------------------------------------------------------------------------------
if [ "$INSTALL_MODE" = "system" ]; then
    TARGET_DIR="$SYSTEM_INSTALL_DIR"
    BIN_DIR="$SYSTEM_BIN_DIR"
else
    TARGET_DIR="$USER_INSTALL_DIR"
    BIN_DIR="$USER_BIN_DIR"
fi

info "Setting up runtime directory at $TARGET_DIR..."
if [ "$INSTALL_MODE" = "system" ]; then
    sudo mkdir -p "$TARGET_DIR"
    sudo chown -R "$USER":"$USER" "$TARGET_DIR"
else
    mkdir -p "$TARGET_DIR"
fi

VENV_PATH="$TARGET_DIR/venv"
if [ ! -d "$VENV_PATH" ]; then
    info "Creating dedicated Python virtual environment at $VENV_PATH..."
    $PYTHON_BIN -m venv "$VENV_PATH"
fi

info "Installing optimized OpenVINO GenAI runtime and iNPU components..."
"$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV_PATH/bin/pip" install -e "$REPO_DIR" -q

success "Installed iNPU core engine into $VENV_PATH"

# ------------------------------------------------------------------------------
# 5. Executable Wrapper Creation
# ------------------------------------------------------------------------------
info "Creating unified CLI binary wrapper: $CMD_NAME..."

mkdir -p "$BIN_DIR" 2>/dev/null || sudo mkdir -p "$BIN_DIR"

WRAPPER_PATH="$BIN_DIR/$CMD_NAME"
WRAPPER_SCRIPT="#!/usr/bin/env bash
exec \"$VENV_PATH/bin/python\" -m intelflow.cli \"\$@\"
"

if [ "$INSTALL_MODE" = "system" ]; then
    echo "$WRAPPER_SCRIPT" | sudo tee "$WRAPPER_PATH" >/dev/null
    sudo chmod +x "$WRAPPER_PATH"
else
    echo "$WRAPPER_SCRIPT" > "$WRAPPER_PATH"
    chmod +x "$WRAPPER_PATH"
    if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
        warn "$USER_BIN_DIR is not in your current \$PATH. Add it by running:"
        warn "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    fi
fi

success "Binary installed to $WRAPPER_PATH"

# ------------------------------------------------------------------------------
# 6. Systemd User Service (Optional)
# ------------------------------------------------------------------------------
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
if [ -f "$REPO_DIR/inpu.service" ]; then
    sed "s|/usr/bin/inpu|$WRAPPER_PATH|g" "$REPO_DIR/inpu.service" > "$SYSTEMD_USER_DIR/inpu.service"
    success "Installed systemd user service to $SYSTEMD_USER_DIR/inpu.service"
    info "To enable automatic background server: systemctl --user enable --now inpu.service"
fi

# ------------------------------------------------------------------------------
# 7. Verification
# ------------------------------------------------------------------------------
echo ""
info "Verifying iNPU installation..."
"$WRAPPER_PATH" validate || true

echo ""
cat << EOF
================================================================================
🎉 ${BOLD}${APP_NAME} installed successfully!${NC}
================================================================================

Quick start:
  ${CYAN}inpu run qwen2.5${NC}       # Interactive terminal chat on Intel NPU
  ${CYAN}inpu top${NC}               # Live real-time hardware telemetry monitor
  ${CYAN}inpu serve qwen2.5${NC}     # Start dual OpenAI/Ollama API server (port 52625)
  ${CYAN}inpu bench qwen2.5${NC}     # Benchmark throughput & latency on NPU
  ${CYAN}inpu --help${NC}            # View full command reference

For AI coding agents (Antigravity, Claude Code, Cursor, Aider):
  See ${BOLD}AGENTS.md${NC} for Model Context Protocol (MCP) integration.
================================================================================
EOF
