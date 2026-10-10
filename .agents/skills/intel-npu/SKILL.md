---
name: intel-npu
description: Offloads local micro-tasks, code diff summaries, symbol explanations, and vector embeddings to the onboard Intel NPU (Core Ultra Meteor Lake) via the intel-npu MCP server, saving cloud token quota with zero fan noise.
---

# Intel NPU Local Coprocessor Skill

This skill guides AI coding agents (**Google Antigravity**, **Claude Code**, **Cursor**, **Open Code**, **Aider**) to utilize the workstation's local **Intel Core Ultra NPU** (Neural Processing Unit) via the `intel-npu` MCP server.

---

## ⚡ Hardware & Empirical Performance Profile

Tested on **Intel Core Ultra 7 155H** (`Intel(R) AI Boost`, Arch: 3720, OpenVINO INT4/INT8):

| Tool | Model & Precision | Measured Latency | Power / Fan | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`embed_npu`** | `bge-small` (INT8, 384-d) | **~200ms / item** (~1.0s for 5 chunks) | ~2.5W / Fanless | **⭐ Killer Feature**: Repository indexing & vector search |
| **`npu_status`** | Hardware sysfs / Level Zero | **< 2ms** | 0W overhead | Silicon health, memory, and telemetry monitoring |
| **`ask_npu`** | `qwen2.5` 1.5B (INT4) | **~7s** (~100 tokens) | ~2.5W / Fanless | Docstrings, syntax lookups, boilerplate, offline mode |
| **`explain_symbol_npu`** | `qwen2.5` 1.5B (INT4) | **~16s** (~150 tokens) | ~2.5W / Fanless | Deep-dive explanations of functions, classes, and types |
| **`summarize_code_npu`** | `qwen2.5` 1.5B (INT4) | **~23s** (30–60 line diff) | ~2.5W / Fanless | PR summaries, changelog entries, commit message drafts |

---

## 🛠️ Tool Recipes & Best Practices

### 1. `embed_npu(texts: list[str])` — Always Offload Locally
- **Best For**: Indexing repository files, symbol search, code search.
- **Why**: $0 cloud API cost, 100% workstation privacy, ~200ms per text item.
- **Tip**: Batch text items in chunks of 16–32 items for optimal memory throughput.

### 2. `npu_status()` — Hardware Health & Telemetry
- **Best For**: Session startup verification, verifying active model weights, checking NPU power state (`D0`/`D3hot`).
- **Telemetry Fields**: `busy_time_sec`, `current_freq_mhz`, `memory_mb`, `power_state`.

### 3. `summarize_code_npu(code: str, language: str)` — Background & Artifact Tasks
- **Best For**: Generating commit message summaries, PR descriptions, and changelogs.
- **Prompting Tip**: Pass focused git diffs or functions ($\le 2,000$ tokens) for clean, structured analysis.

### 4. `explain_symbol_npu(symbol: str, context: str)` — Targeted Deep-Dives
- **Best For**: Explaining complex third-party or internal API symbols without sending proprietary code to the cloud.

### 5. `ask_npu(prompt: str, system_prompt: str)` — Offline & Boilerplate Generation
- **Best For**: Generating function docstrings, stub implementations, mock fixtures, and functioning when disconnected from the internet.

---

## 🧠 Workload Architecture by Agent Platform

- **Google Antigravity**: Uses `call_mcp_tool("intel-npu", ...)` to offload vector indexing and background doc generation while maintaining fast cloud frontier interactive turns.
- **Claude Code**: Exposes `inpu` via stdio MCP in `~/.config/Claude/claude_desktop_config.json`.
- **Cursor & Open Code**: Configured via `.cursor/mcp.json` or `.open-code/mcp.json`.
- **Aider**: Can attach via background REST endpoint: `inpu serve qwen2.5 &` and `aider --openai-api-base http://127.0.0.1:52625/v1`.
