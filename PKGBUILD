# Maintainer: wen
pkgname=intelflowlm
pkgver=0.1.0
pkgrel=1
pkgdesc="iNPU: Ultra-efficient Intel NPU inference engine, CLI, and dual OpenAI/Ollama server (FastFlowLM counterpart)"
arch=('x86_64')
url="https://github.com/wen/iNPU"
license=('MIT')
depends=(
    'intel-npu-driver'
    'intel-npu-compiler'
    'openvino-intel-npu-plugin'
    'level-zero-loader'
    'python'
    'python-openvino'
    'python-fastapi'
    'uvicorn'
    'python-rich'
    'python-typer'
    'python-huggingface-hub'
    'python-httpx'
    'python-numpy'
)
makedepends=(
    'python-build'
    'python-installer'
    'python-wheel'
    'python-setuptools'
)
optdepends=(
    'openvino-genai-bin: Precompiled OpenVINO GenAI runtime for NPU LLMPipeline'
)
provides=('inpu')
conflicts=('intelflowlm-git')
source=(
    "inpu.service"
    "LICENSE"
    "99-intelflow-accel.rules"
)
sha256sums=(
    '3fecdebf9ff85ffab427191a7a1c486e17a3725badd16c4b3d15f1adc7ff7169'
    '32431c1faa14d04e1c617aed30e05453de6517e1c558ade4479fea48b6509a7f'
    '03dbbe83656cf1d6e0bb36bb058c3b2e429135dc3f8863e581177a3b75e9a43f'
)

build() {
    cd "$startdir"
    local _py="python"
    if ! $_py -c "import setuptools, build" &>/dev/null && [ -x "$startdir/.venv/bin/python" ]; then
        _py="$startdir/.venv/bin/python"
    fi

    $_py -m build --wheel --no-isolation
}

package() {
    cd "$startdir"
    local _py="python"
    if ! $_py -c "import installer" &>/dev/null && [ -x "$startdir/.venv/bin/python" ]; then
        _py="$startdir/.venv/bin/python"
    fi

    $_py -m installer --destdir="$pkgdir" --prefix=/usr dist/*.whl

    # Normalize shebang to Arch system python across all entrypoints
    sed -i '1s|^#!.*python.*|#!/usr/bin/python|' "$pkgdir/usr/bin/"*

    # Systemd user unit and udev rules
    install -Dm644 "$srcdir/inpu.service" "$pkgdir/usr/lib/systemd/user/inpu.service"
    install -Dm644 "$srcdir/99-intelflow-accel.rules" "$pkgdir/usr/lib/udev/rules.d/99-intelflow-accel.rules"

    # License and documentation
    install -Dm644 "$srcdir/LICENSE" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
}
