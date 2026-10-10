# Mandatory Local Offloading to Intel NPU

When executing development tasks in this workspace, the agent MUST proactively offload the following micro-tasks to the local Intel NPU via the `intel-npu` MCP server (`call_mcp_tool`):

1. **Git Diff & Change Summaries**:
   - Always call `summarize_code_npu(code=diff, language="diff")` before writing changelog summaries or pull request drafts.
2. **Symbol & API Explanations**:
   - When asked to explain a symbol, class, method, or struct, invoke `explain_symbol_npu(symbol=name, context=code)`.
3. **Semantic Embeddings & Indexing**:
   - Use `embed_npu(texts=[...])` for vector transformations instead of any cloud embedding APIs.
4. **Offline / Boilerplate Generation**:
   - Delegate repetitive docstrings, boilerplate, or unit test stubs to `ask_npu(prompt=...)`.
5. **Silicon Telemetry**:
   - Check `npu_status()` to monitor local power and memory usage.

DO NOT waste cloud token quota or introduce unnecessary network latency for these tasks when the Intel NPU is available.
