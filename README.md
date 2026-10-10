<p align="center">
  <h1 align="center">⚡ iNPU</h1>
  <p align="center"><b>Unlock Intel® Core™ Ultra NPUs — Ultra-Efficient Local AI on Linux</b></p>
  <p align="center">
    <i>FastFlowLM counterpart built exclusively for Intel AI Boost (Meteor Lake & Core Ultra processors).</i>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-Intel%20AI%20Boost%20(NPU%203720)-blue?style=flat-square" alt="Hardware" />
  <img src="https://img.shields.io/badge/Efficiency-~2.5W%20Power%20Profile-green?style=flat-square" alt="Efficiency" />
  <img src="https://img.shields.io/badge/Throughput-43%2B%20tok%2Fs%20(Qwen%202.5)-orange?style=flat-square" alt="Throughput" />
  <img src="https://img.shields.io/badge/API-OpenAI%20%2B%20Ollama%20Dual%20Mode-purple?style=flat-square" alt="API" />
  <img src="https://img.shields.io/badge/Agent%20Ready-Model%20Context%20Protocol%20(MCP)-teal?style=flat-square" alt="MCP" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License" />
</p>

---

## 💡 What is iNPU?

**iNPU** is an open-source, NPU-first inference runtime, developer toolkit, and local API server designed to unlock the onboard **Intel AI Boost Neural Processing Unit (Meteor Lake, Arrow Lake, Lunar Lake)** on Linux.

Inspired by AMD's **FastFlowLM (`flm`)** and **Ollama**, `iNPU` provides a single unified CLI (`inpu`) that offloads large language models, vision models, and semantic embeddings directly to dedicated NPU silicon — with **zero discrete GPU requirement**, **flat CPU utilization (< 5%)**, and **whisper-quiet fanless execution (~2.5W power draw)**.

---

## ⚡ Highlights

- 🧠 **100% NPU Silicon Execution**: Binds directly to `/dev/accel/accel0` through the Linux DRM `intel_vpu` driver and Level Zero compiler.
- 🚀 **High Throughput**: Delivers **43+ tokens/sec** decode throughput and **~220 ms TTFT** on Intel Core Ultra 7 155H.
- 🔋 **Extreme Energy Efficiency**: Consumes **~2.5W** (vs 35W+ on CPU) — run local LLMs all day on laptop battery power without warming your lap or spinning fans.
- 🔌 **Dual OpenAI & Ollama REST Server**: Listens on port `52625` with full compatibility for **Open WebUI**, **Obsidian**, **LangChain**, **Aider**, and **LiteLLM**.
- 📊 **Real-time Silicon Telemetry (`inpu top`)**: Built-in terminal hardware monitor inspecting kernel sysfs clock frequency (0–1400 MHz), power state (`D0`/`D3hot`), and memory allocation.
- 🤖 **Native AI Coding Agent Toolkit (`AGENTS.md`)**: Full **Model Context Protocol (MCP)** stdio server (`inpu mcp`) allowing agents (Antigravity, Claude Code, Cursor) to offload local micro-tasks at zero cloud token cost.

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User & Client Ecosystem                         │
│  Terminal REPL  │  Open WebUI  │  Obsidian  │  LangChain  │  AI Agents │
└────────┬────────┴──────┬───────┴─────┬──────┴──────┬──────┴─────┬──────┘
         │               │             │             │            │
         ▼               ▼             ▼             ▼            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           iNPU Unified Core                            │
│   CLI: inpu run   │   REST: /v1/* (OpenAI) & /api/* (Ollama)  │  MCP   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        OpenVINO GenAI Subsystem                        │
│         LLMPipeline (INT4 KV-Cache)  │  TextEmbeddingPipeline          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Level Zero (libze_loader.so)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Linux Kernel & Intel Driver                        │
│          /dev/accel/accel0 (intel_vpu Linux DRM Driver Node)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Physical Compute
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Intel® AI Boost NPU Silicon (Meteor Lake 3720)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Comparison

| Feature | **iNPU** (This Project) | **FastFlowLM** (ROCm) | **Ollama** |
| :--- | :--- | :--- | :--- |
| **Target Hardware** | **Intel Core Ultra NPU** | AMD Ryzen AI NPU | CPU / NVIDIA / AMD GPU |
| **CLI Binary** | Single **`inpu`** | `flm` | `ollama` |
| **Power Consumption** | **~2.5W** (Ultra-low) | ~3W | 30W – 150W+ |
| **Fan Noise** | **Silent (0 fan noise)** | Silent | Loud / High RPM |
| **Dual REST Server** | **OpenAI + Ollama** | OpenAI + Ollama | Ollama native |
| **Hardware Telemetry** | **`inpu top` (Live DRM sysfs)**| Task Manager NPU tab | None built-in |
| **AI Agent Support** | **MCP Stdio Server (`inpu mcp`)**| Standalone only | None built-in |
| **Interactive REPL** | **11 Slash Commands** | 11 Slash Commands | Minimal |

---

## 🚀 Quick Start

### Option 1: Universal Installer (Recommended)

Run the automated installer to configure drivers, permissions, and binary wrappers:

```bash
git clone https://github.com/wen/iNPU.git
cd iNPU
./install.sh
```

*(Or user-space only without requiring root: `./install.sh --user`)*

### Option 2: Arch Linux (PKGBUILD)

```bash
makepkg -si
```

---

## 💻 CLI Command Reference

`iNPU` provides a single unified command: **`inpu`**.

### 1. Interactive Chat (`inpu run`)

Launch an interactive streaming REPL on the NPU:

```bash
inpu run qwen2.5
```

#### Interactive Slash Commands
While chatting in `inpu run`, you have access to 11 built-in slash commands:

| Command | Action |
| :--- | :--- |
| **`/?`** or **`/help`** | Display interactive command cheat sheet |
| **`/show`** | Display active model architecture, context limit, and device details |
| **`/clear`** | Flush conversation history and reset NPU KV-cache memory |
| **`/history`** | Pretty-print turn-by-turn conversation history |
| **`/status`** | Display last latency, TTFT, token throughput, and token counts |
| **`/verbose`** | Toggle per-turn hardware telemetry reporting |
| **`/load <model>`** | Dynamically switch models without exiting the session |
| **`/save [file]`** | Save chat transcript to a markdown/text file |
| **`/input "<file>" [prompt]`** | Load external plain text documents into your prompt |
| **`/set <param> <val>`** | Adjust hyperparameters (`gen-lim`, `temp`, `top-p`) |
| **`/bye`** or **`/exit`** | Cleanly exit the session |

---

### 2. Live Hardware Telemetry Monitor (`inpu top`)

Monitor real-time NPU hardware counters directly from the Linux DRM kernel driver:

```bash
inpu top
# Or single snapshot:
inpu top --once
```

**Live Output:**
```
╭──────────────────── ⚡ IntelFlowLM NPU Hardware Monitor ─────────────────────╮
│  Device                  Intel(R) AI Boost (NPU    Meteor Lake / Core Ultra  │
│                          3720)                                               │
│  Compute Utilization     0.0%                      Silicon active duty       │
│                                                    cycle                     │
│  Current Clock           700 MHz / 1400 MHz        Dynamic frequency         │
│                                                    scaling                   │
│  Power State             D0 (Active Compute)       Ultra-low power (~2-4W)   │
│  Memory Allocated        65.54 MB                  68,722,688 bytes          │
│  Cumulative Compute      141.47 seconds            141,469,021 µs hardware   │
│                                                    counter                   │
│                                                                              │
│                    Active Processes on /dev/accel/accel0                     │
│  PID        ┃ Process            ┃ Command Line                              │
│ ━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  70283      │ inpu               │ python -m intelflow.cli run qwen2.5       │
╰──────────────────────────── Press Ctrl+C to exit ────────────────────────────╯
```

---

### 3. Dual REST API Server (`inpu serve`)

Start the background server on port `52625` (FastFlowLM default):

```bash
inpu serve qwen2.5
```

- **OpenAI Standard Endpoints**:
  - `GET  /v1/models`
  - `GET  /v1/version`
  - `POST /v1/chat/completions` (supports real-time SSE streaming)
  - `POST /v1/completions` (legacy text completions)
  - `POST /v1/embeddings` (vector embeddings via `bge-small`)
- **Ollama Standard Endpoints**:
  - `GET  /api/tags` (model list for Open WebUI)
  - `GET  /api/version`
  - `GET  /api/ps` (running models & VRAM allocations)
  - `GET  /api/npu/status` (live silicon hardware status)
  - `POST /api/chat` (streaming NDJSON)
  - `POST /api/generate`
  - `POST /api/embeddings`
  - `POST /api/pull`

---

### 4. Standardized Hardware Benchmark (`inpu bench`)

Measure Time-to-First-Token (TTFT) and decode throughput directly on NPU silicon:

```bash
inpu bench qwen2.5 --tokens 64
```

```
                               Benchmark Results                                
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric                          ┃ Result                    ┃ Unit           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Device                          │ NPU                       │ -              │
│ Time to First Token (TTFT)      │ 224.02                    │ ms             │
│ Decode Throughput               │ 43.08                     │ tokens / sec   │
│ Tokens Generated                │ 32                        │ tokens         │
│ Total Generation Time           │ 0.944                     │ seconds        │
└─────────────────────────────────┴───────────────────────────┴────────────────┘
```

---

### 5. Model Management Commands

```bash
# Pull model from Hugging Face
inpu pull qwen2.5
inpu pull llama3.2:1b --force

# List cached and catalog models
inpu list
inpu list --filter installed
inpu list --json

# Verify integrity of downloaded model files
inpu check qwen2.5

# Delete model from cache
inpu remove qwen2.5

# Check server port
inpu port

# Validate system readiness and DRM permissions
inpu validate --json
```

---

## 🔌 Client Ecosystem Integration

Because `iNPU` exposes both OpenAI and Ollama protocols on `http://127.0.0.1:52625`, connecting existing local AI apps is effortless:

### Open WebUI
1. Go to **Settings** $\to$ **Connections** $\to$ **Ollama API**.
2. Set URL to: `http://127.0.0.1:52625`.
3. Your downloaded models appear automatically in the dropdown!

### Obsidian (Local REST / Smart Connections)
1. Set API base URL to: `http://127.0.0.1:52625/v1`.
2. Model: `qwen2.5` or `bge-small` (for embeddings).

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="inpu")

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "Explain quantum superposition."}],
)
print(response.choices[0].message.content)
```

---

## 🤖 AI Agent Integration (`AGENTS.md`)

`iNPU` features native **Model Context Protocol (MCP)** integration over stdio:

```bash
inpu mcp
```

This unlocks offline, zero-token micro-task delegation for **Google Antigravity**, **Claude Code**, **Cursor**, and **Aider**. 

See [**`AGENTS.md`**](./AGENTS.md) for full configuration snippets and tool specifications (`ask_npu`, `summarize_code_npu`, `explain_symbol_npu`, `embed_npu`, `npu_status`).

---

## 📦 Verified Model Catalog

All models are pre-quantized in INT4/INT8 OpenVINO IR format and verified on Meteor Lake NPU:

| Model Tag | Parameters | Format | Context Window | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`qwen2.5`** *(Default)* | 1.5B | INT4 | 4096 | Fast code analysis, reasoning (43+ tok/s) |
| **`qwen2.5:3b`** | 3B | INT4 | 4096 | Higher accuracy multi-turn reasoning |
| **`qwen2.5:7b`** | 7B | INT4 | 4096 | High reasoning accuracy (systems with 32GB+ RAM) |
| **`llama3.2:1b`** | 1B | INT4 | 2048 | Lightweight conversational tasks |
| **`llama3.2:3b`** | 3B | INT4 | 2048 | General purpose assistant |
| **`gemma2:2b`** | 2B | INT4 | 4096 | Instruction following |
| **`phi3.5:mini`** | 3.8B | INT4 | 4096 | Logical reasoning and synthesis |
| **`bge-small`** | 33M | INT8 | 512 | Rapid semantic vector search & embeddings |

---

## 📄 License & Credits

- Licensed under the [MIT License](./LICENSE).
- Inspired by [FastFlowLM (ROCm)](https://github.com/ROCm/FastFlowLM) and [Ollama](https://github.com/ollama/ollama).
- Powered by the **Intel® AI Boost NPU Architecture** and **OpenVINO™ GenAI**.
