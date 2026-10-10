"""
Configuration settings and Model Registry for iNPU.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Directories
HOME_DIR = Path.home()
CONFIG_DIR = Path(os.getenv("INPU_CONFIG_DIR", os.getenv("INTELFLOW_CONFIG_DIR", os.getenv("FLM_CONFIG_DIR", HOME_DIR / ".config" / "inpu"))))
CACHE_DIR = Path(os.getenv("INPU_CACHE_DIR", os.getenv("INTELFLOW_CACHE_DIR", HOME_DIR / ".cache" / "inpu")))
MODEL_DIR = Path(os.getenv("INPU_MODEL_PATH", os.getenv("INTELFLOW_MODEL_PATH", os.getenv("FLM_MODEL_PATH", CACHE_DIR / "models"))))
BLOB_CACHE_DIR = Path(os.getenv("INPU_BLOB_CACHE", os.getenv("INTELFLOW_BLOB_CACHE", CACHE_DIR / "blob_cache")))

for d in [CONFIG_DIR, CACHE_DIR, MODEL_DIR, BLOB_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default Server Settings
DEFAULT_HOST = os.getenv("INPU_HOST", os.getenv("INTELFLOW_HOST", os.getenv("FLM_HOST", "127.0.0.1")))
DEFAULT_PORT = int(os.getenv("INPU_PORT", os.getenv("INTELFLOW_PORT", os.getenv("FLM_PORT", 52625))))
DEFAULT_MODEL = os.getenv("INPU_MODEL", os.getenv("INTELFLOW_MODEL", os.getenv("FLM_MODEL", "qwen2.5")))

# Verified NPU-Compatible Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen2.5": {
        "repo_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "description": "Qwen 2.5 1.5B Instruct (INT4) - Fast reasoning, excellent code support (Default)",
        "context_length": 4096,
        "type": "llm",
    },
    "qwen2.5:1.5b": {
        "repo_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "description": "Qwen 2.5 1.5B Instruct (INT4)",
        "context_length": 4096,
        "type": "llm",
    },
    "qwen2.5:0.5b": {
        "repo_id": "llmware/qwen2.5-0.5b-instruct-ov",
        "description": "Qwen 2.5 0.5B Instruct (INT4) - Ultra-lightweight",
        "context_length": 2048,
        "type": "llm",
    },
    "qwen2.5:3b": {
        "repo_id": "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov",
        "description": "Qwen 2.5 Coder 3B Instruct (INT4) - Dedicated code generation",
        "context_length": 4096,
        "type": "llm",
    },
    "qwen2.5:7b": {
        "repo_id": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
        "description": "Qwen 2.5 7B Instruct (INT4) - Frontier accuracy for 32GB+ systems",
        "context_length": 4096,
        "type": "llm",
    },
    "llama3.2:1b": {
        "repo_id": "llmware/llama-3.2-1b-instruct-npu-ov",
        "description": "Meta Llama 3.2 1B Instruct (INT4 NPU-optimized)",
        "context_length": 2048,
        "type": "llm",
    },
    "llama3.2:3b": {
        "repo_id": "srang992/Llama-3.2-3B-Instruct-ov-INT4",
        "description": "Meta Llama 3.2 3B Instruct (INT4)",
        "context_length": 2048,
        "type": "llm",
    },
    "gemma2:2b": {
        "repo_id": "OpenVINO/gemma-2b-it-int4-ov",
        "description": "Google Gemma 2B Instruct (INT4)",
        "context_length": 4096,
        "type": "llm",
    },
    "phi3.5:mini": {
        "repo_id": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
        "description": "Microsoft Phi 3.5 Mini Instruct (INT4)",
        "context_length": 4096,
        "type": "llm",
    },
    "smollm2:1.7b": {
        "repo_id": "AIFunOver/SmolLM2-1.7B-Instruct-openvino-4bit",
        "description": "SmolLM2 1.7B Instruct (INT4)",
        "context_length": 4096,
        "type": "llm",
    },
    "deepseek-r1:1.5b": {
        "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
        "description": "DeepSeek R1 Distill Qwen 1.5B (INT4) - Step-by-step reasoning",
        "context_length": 4096,
        "type": "llm",
    },
    "bge-small": {
        "repo_id": "laion/bge-small-en-v1.5_openvino_int8",
        "description": "BAAI BGE Small EN v1.5 (INT8) - Fast code/text embeddings (384-d)",
        "type": "embedding",
    },
    "bge-base": {
        "repo_id": "OpenVINO/bge-base-en-v1.5-int8-ov",
        "description": "BAAI BGE Base EN v1.5 (INT8) - Intel official embeddings (768-d)",
        "type": "embedding",
    },
}

def resolve_model_tag(tag: str) -> str:
    """Normalize model tag (e.g. 'qwen2.5' -> registry repo or direct repo)."""
    raw_tag = tag.strip()
    lower_tag = raw_tag.lower()
    if lower_tag in MODEL_REGISTRY:
        return MODEL_REGISTRY[lower_tag]["repo_id"]
    return raw_tag
