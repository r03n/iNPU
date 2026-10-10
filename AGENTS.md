# 🤖 iNPU for AI Coding Agents

> **Offload repetitive micro-tasks to your laptop's Intel NPU with zero cloud token cost, zero battery drain, and zero fan noise.**

---

## 🎯 Overview

Autonomous coding agents (such as **Google Antigravity**, **Claude Code**, **Cursor**, and **Aider**) frequently perform routine, token-hungry micro-tasks:
- Summarizing large git diffs before creating pull requests
- Explaining functions, classes, and types
- Generating docstrings and inline comments
- Generating semantic vector embeddings for codebase indexing
- Drafting commit messages and issue templates

Routing these micro-tasks to cloud frontier models (Gemini 2.5 Pro, Claude 3.5 Sonnet, GPT-4o) wastes cloud rate quotas, increases round-trip latency, and incurs ongoing API costs.

**iNPU** solves this by providing a native **Model Context Protocol (MCP)** stdio server that runs directly on your **Intel Core Ultra NPU**. Agents can transparently offload micro-tasks to local silicon with:
- 💸 **$0 API Cost**: Free, unlimited local inference.
- 🔐 **100% Local Privacy**: Proprietary code never leaves your workstation.
- 🔋 **~2.5W Power Profile**: Silent, fanless execution that preserves laptop battery life.
- ⚡ **Zero Cloud Latency**: 43+ tokens/sec on-device decode speed.

---

## 🧠 Workload Routing Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AI Coding Agent Request                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  Is this a complex architecture task?
                                   / \
                                  /   \
                             Yes /     \ No (Micro-task)
                                /       \
                               ▼         ▼
┌───────────────────────────────────┐   ┌────────────────────────────────┐
│         Cloud Frontier LLM        │   │         Local Intel NPU        │
│  (Gemini 2.5 Pro / Claude 3.5)    │   │      (via iNPU MCP Server)     │
├───────────────────────────────────┤   ├────────────────────────────────┤
│ • Multi-file architectural plans  │   │ • Git diff & pull request docs │
│ • Deep multi-step reasoning       │   │ • Function / symbol summaries  │
│ • Complex bug debugging           │   │ • Vector code embeddings       │
│ • High-level design trade-offs    │   │ • Boilerplate / docstrings     │
└───────────────────────────────────┘   └────────────────────────────────┘
```

---

## 🛠️ MCP Tool Reference

When running `inpu mcp`, the following tools are exposed over standard input/output (stdio):

### 1. `ask_npu`
- **Description**: Query the local Intel NPU LLM (`qwen2.5` 1.5B INT4) for drafting, formatting, or offline generation.
- **Parameters**:
  - `prompt` (*string*, required): User instruction or text.
  - `system_prompt` (*string*, optional): Custom system instructions.
  - `max_tokens` (*integer*, optional, default `512`): Generation length.

### 2. `summarize_code_npu`
- **Description**: Summarize a code block, function, or git diff locally on the Intel NPU.
- **Parameters**:
  - `code` (*string*, required): The code snippet, function, or diff content.
  - `language` (*string*, optional): Programming language (e.g. `python`, `rust`, `c++`).

### 3. `explain_symbol_npu`
- **Description**: Explain a specific symbol, class, struct, function, or type in its local context.
- **Parameters**:
  - `symbol` (*string*, required): Name of the symbol to explain.
  - `context` (*string*, optional): Surrounding code or declaration context.

### 4. `embed_npu`
- **Description**: Generate normalized 384-dimensional vector embeddings using the local NPU (`bge-small`) for rapid semantic repository indexing and search.
- **Parameters**:
  - `texts` (*list of strings*, required): List of document or code strings to embed.
- **Returns**: Object containing `count` (*int*), `dimensions` (*int*, e.g. 384), and `embeddings` (*List[List[float]]*).

### 5. `npu_status`
- **Description**: Query real-time silicon hardware health, current clock frequency (MHz), power state (`D0`/`D3hot`), memory usage, and cumulative compute time.
- **Parameters**: None.

---

## ⚙️ Agent Configuration Guides

### 1. Google Antigravity

To make iNPU globally available across all workspaces in Google Antigravity:

Add to `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "intel-npu": {
      "command": "/usr/local/bin/inpu",
      "args": ["mcp"]
    }
  }
}
```

Or for project-level workspace configuration, add to `.agents/mcp_config.json`:
```json
{
  "mcpServers": {
    "intel-npu": {
      "command": "inpu",
      "args": ["mcp"]
    }
  }
}
```

---

### 2. Claude Desktop / Claude Code

Add to `~/.config/Claude/claude_desktop_config.json` (Linux):
```json
{
  "mcpServers": {
    "inpu": {
      "command": "/usr/local/bin/inpu",
      "args": ["mcp"]
    }
  }
}
```

---

### 3. Cursor & Open Code / Windsurf

Add to `.cursor/mcp.json`, `.open-code/mcp.json`, or `.codeium/windsurf/mcp_config.json`:
```json
{
  "mcpServers": {
    "inpu": {
      "command": "inpu",
      "args": ["mcp"]
    }
  }
}
```

---

### 4. Aider CLI

Connect Aider directly to iNPU's OpenAI-compatible background server:

```bash
# 1. Start iNPU server in background
inpu serve qwen2.5 &

# 2. Launch Aider with local NPU endpoint
aider --openai-api-base http://127.0.0.1:52625/v1 --openai-api-key inpu --model openai/qwen2.5
```

---

## ⚡ Empirical Benchmark Profile (Intel AI Boost Meteor Lake)

Verified hardware benchmarks on **Intel Core Ultra 7 155H** (OpenVINO INT4/INT8):

| Tool | Precision | Measured Latency | Power Profile | Optimal Agent Role |
| :--- | :--- | :--- | :--- | :--- |
| **`embed_npu`** | INT8 (384-d) | **~200ms / item** (~1.0s for 5 texts) | ~2.5W / Fanless | **⭐ Killer Feature**: Codebase semantic search & vector indexing |
| **`npu_status`** | Sysfs / L0 | **< 2ms** | 0W overhead | Device readiness & hardware load telemetry |
| **`ask_npu`** | INT4 (1.5B) | **~7s** (~100 tokens) | ~2.5W / Fanless | Docstrings, syntax lookups, offline generation |
| **`explain_symbol_npu`** | INT4 (1.5B) | **~16s** (~150 tokens) | ~2.5W / Fanless | Function/class/type analysis without cloud egress |
| **`summarize_code_npu`** | INT4 (1.5B) | **~23s** (30–60 line diff) | ~2.5W / Fanless | Background PR drafts, changelog summaries |

---

## 📈 Best Practices & Smart Hybrid Routing

1. **Check NPU Readiness First**: Agents should invoke `npu_status()` at session startup to confirm device readiness and active models.
2. **Prioritize Local Embeddings**: Always delegate semantic codebase indexing to `embed_npu(texts=[...])`. The NPU processes dense matrix transformations with zero CPU overhead, saving significant API token costs.
3. **Smart Hybrid Routing**:
   - **Always Offload**: Vector embeddings and hardware diagnostics.
   - **Offload for Micro-Tasks & Background Runs**: Git diff summaries (`summarize_code_npu`), symbol explanations (`explain_symbol_npu`), and docstrings (`ask_npu`).
   - **Retain on Cloud Frontier**: Interactive high-speed chat, complex multi-file planning, and heavy refactoring where sub-second turn latency is essential.
4. **Graceful Fallback**: If an agent requests a sequence length exceeding hardware capacity, iNPU cleanly falls back without crashing the agent session.


