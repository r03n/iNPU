"""
Model Context Protocol (MCP) Server for IntelFlowLM.
Exposes local Intel NPU tools to Antigravity over stdio.
"""

import sys
from typing import List, Dict, Any, Optional
from mcp.server.mcpserver import MCPServer

from intelflow.engine import NPUEngine
from intelflow.embeddings import NPUEmbeddingEngine
from intelflow.config import DEFAULT_MODEL

mcp = MCPServer("intel-npu")

_engine: Optional[NPUEngine] = None
_embed_engine: Optional[NPUEmbeddingEngine] = None

def get_engine() -> NPUEngine:
    global _engine
    if _engine is None:
        _engine = NPUEngine(DEFAULT_MODEL, device="NPU")
    return _engine

def get_embed_engine() -> NPUEmbeddingEngine:
    global _embed_engine
    if _embed_engine is None:
        _embed_engine = NPUEmbeddingEngine("bge-small")
    return _embed_engine

@mcp.tool()
def ask_npu(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    """
    Query the local Intel NPU LLM (Qwen 2.5 1.5B INT4) with zero cloud latency and zero API cost.
    Use for drafting, fast answers, boilerplate generation, or offline tasks.
    """
    engine = get_engine()
    full_prompt = f"<|im_start|>system\n{system_prompt or 'You are a helpful coding assistant running locally on Intel NPU.'}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    res = engine.generate(full_prompt, max_new_tokens=max_tokens)
    return res["text"]

@mcp.tool()
def summarize_code_npu(code: str, language: str = "") -> str:
    """
    Summarize a code block, function, or git diff locally on the Intel NPU.
    Saves cloud frontier token quota and operates completely privately.
    """
    engine = get_engine()
    prompt = (
        f"<|im_start|>system\nYou are an expert code analyst. Provide a clear, concise bulleted summary of this code.<|im_end|>\n"
        f"<|im_start|>user\nLanguage: {language or 'auto'}\n\n```{language}\n{code}\n```\n"
        f"Summarize the logic, key components, and potential side-effects:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    res = engine.generate(prompt, max_new_tokens=350)
    return res["text"]

@mcp.tool()
def explain_symbol_npu(symbol: str, context: str = "") -> str:
    """
    Explain a specific code symbol, class, function, or type using local context on the Intel NPU.
    """
    engine = get_engine()
    prompt = (
        f"<|im_start|>system\nExplain the specified symbol and its purpose clearly and concisely.<|im_end|>\n"
        f"<|im_start|>user\nSymbol: {symbol}\nContext:\n{context}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    res = engine.generate(prompt, max_new_tokens=250)
    return res["text"]

@mcp.tool()
def embed_npu(texts: List[str]) -> List[List[float]]:
    """
    Generate normalized semantic vector embeddings for a list of text strings using the local Intel NPU.
    Enables rapid semantic code search and symbol indexing.
    """
    embed_engine = get_embed_engine()
    return embed_engine.embed_documents(texts)

@mcp.tool()
def npu_status() -> Dict[str, Any]:
    """
    Check the current hardware status of the Intel NPU, loaded models, and device metrics.
    """
    from intelflow.monitor import NPUTelemetry
    import openvino as ov
    core = ov.Core()
    npu_present = "NPU" in core.available_devices
    
    device_name = "Intel NPU"
    arch = "Unknown"
    if npu_present:
        try:
            device_name = core.get_property("NPU", "FULL_DEVICE_NAME")
            arch = core.get_property("NPU", "DEVICE_ARCHITECTURE")
        except Exception:
            pass

    telemetry = NPUTelemetry()
    hw_metrics = telemetry.sample() if telemetry.is_npu_available() else {}

    return {
        "device": "NPU",
        "device_name": device_name,
        "architecture": arch,
        "npu_available": npu_present,
        "active_llm": DEFAULT_MODEL,
        "active_embed": "bge-small",
        "power_profile": "Ultra-Low Power (~2-4W)",
        "hardware_telemetry": hw_metrics,
    }

def main():
    mcp.run()

if __name__ == "__main__":
    main()
