---
name: intel-npu
description: Offloads local micro-tasks, code diff summaries, symbol explanations, and vector embeddings to the onboard Intel NPU (Core Ultra Meteor Lake) via the intel-npu MCP server, saving cloud token quota with zero fan noise.
---

# Intel NPU Local Coprocessor Skill

This skill guides Antigravity to utilize the machine's local **Intel Core Ultra NPU** (Neural Processing Unit) via the `intel-npu` MCP server.

## Overview
The Intel NPU draws only ~2–4W of power and runs continuously in the background without waking the discrete GPU or spinning CPU fans. By offloading repetitive, small, or private tasks to the NPU, you save cloud token budget and achieve sub-100ms response times.

## Available Tools

1. **`ask_npu(prompt: str, system_prompt: str)`**:
   - Queries the local NPU model (default: Qwen 2.5 1.5B INT4).
   - Use for: Draft generation, quick syntax lookups, docstring generation, boilerplate snippets.
2. **`summarize_code_npu(code: str, language: str)`**:
   - Generates a concise summary of a code snippet or git diff.
   - Use for: Summarizing long files before editing, compressing large diffs into compact summaries.
3. **`explain_symbol_npu(symbol: str, context: str)`**:
   - Explains a specific class, function, or interface based on provided context.
4. **`embed_npu(texts: list[str])`**:
   - Generates semantic embeddings for code fragments or queries using `bge-small`.
   - Use for: Semantic symbol searching, ranking relevant code snippets.
5. **`npu_status()`**:
   - Returns real-time health, device status, active model, and power metrics.

## Best Practices
- **Hybrid Collaboration**: Use the cloud frontier model (Gemini) for complex architectural planning, multi-file refactors, and debugging, while delegating local micro-tasks (diff summaries, syntax checks) to `intel-npu`.
- **Offline / Private Tasks**: For sensitive proprietary code or when network connectivity is degraded, prioritize `ask_npu`.
