# Smart Hybrid Routing for Intel NPU

To achieve optimal performance, zero cloud token waste, and responsive user interaction, the agent operates under a **Smart Hybrid Routing** model between the cloud frontier orchestrator and the local Intel Core Ultra NPU (`intel-npu` MCP server):

## 1. Always Offload to Intel NPU (Tier 1: Zero Token Waste, High Speed)
- **Codebase Embeddings & Semantic Indexing**:
  - Always invoke `embed_npu(texts=[...])` for vector transformations when indexing files, searching symbols, or ranking context.
  - Never call cloud embedding APIs. Local NPU delivers ~200ms/item latency at ~2.5W with 100% workstation privacy.
- **Hardware Diagnostics & Readiness**:
  - Invoke `npu_status()` to inspect silicon state (`D0`/`D3hot`), memory allocation, and verify device readiness.

## 2. Offload on Demand / Background (Tier 2: Micro-Tasks & Artifacts)
- **Git Diff & Pull Request Documentation**:
  - Delegate diff analysis to `summarize_code_npu(code=diff, language="diff")` when generating PR descriptions, changelog entries, or commit message summaries.
- **Symbol & API Explanations**:
  - Invoke `explain_symbol_npu(symbol=name, context=code)` when the user requests documentation or deep-dives on specific classes, functions, or types.
- **Boilerplate & Docstrings**:
  - Delegate repetitive docstrings, interface stubs, or unit test boilerplate to `ask_npu(prompt=...)`.

## 3. Retain on Cloud Frontier Model (Tier 3: Core Orchestration)
- **High-Speed Interactive Pair Programming**:
  - Maintain fast conversational flow and direct tool operations (reading/writing files, running tests, git commands) on the primary frontier model without inserting unnecessary NPU generation latency into interactive turns.
- **Complex Multi-File Refactoring**:
  - Architectural synthesis, large code modifications, and multi-file debugging belong on the frontier orchestrator.
