"""
Unit tests for ModelHub and configuration.
"""

from intelflow.config import resolve_model_tag, MODEL_REGISTRY
from intelflow.hub import ModelHub

def test_resolve_model_tag():
    assert resolve_model_tag("qwen2.5") == "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
    assert resolve_model_tag("llama3.2:1b") == "llmware/llama-3.2-1b-instruct-npu-ov"
    assert resolve_model_tag("bge-small") == "OpenVINO/bge-small-en-v1.5-int8-ov"
    # Unregistered tag passes through
    assert resolve_model_tag("custom/my-model") == "custom/my-model"

def test_catalog_listing():
    catalog = ModelHub.list_catalog()
    assert "qwen2.5" in catalog
    assert "llama3.2:1b" in catalog
    assert "bge-small" in catalog
    assert catalog["qwen2.5"]["type"] == "llm"
