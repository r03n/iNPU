"""
Embedding generation engine for iNPU using OpenVINO.
Provides text and code embeddings for local semantic search.
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise RuntimeError(
        "OpenVINO GenAI runtime is required for embedding inference on Intel NPU.\n"
        "Please install it using:\n"
        "  pip install openvino-genai --break-system-packages\n"
        "or from AUR:\n"
        "  yay -S openvino-genai-bin"
    ) from e
from intelflow.config import BLOB_CACHE_DIR
from intelflow.hub import ModelHub

class NPUEmbeddingEngine:
    def __init__(
        self,
        model_tag_or_path: str = "bge-small",
        device: str = "NPU",
        cache_dir: Optional[Path] = None,
    ):
        self.device = device.upper()
        self.blob_cache_dir = cache_dir or BLOB_CACHE_DIR

        if os.path.isdir(model_tag_or_path):
            self.model_path = Path(model_tag_or_path)
        else:
            self.model_path = ModelHub.pull(model_tag_or_path)

        # Attempt initialization on NPU, fallback to CPU if model requires dynamic shapes not supported by NPU
        try:
            self.pipe = ov_genai.TextEmbeddingPipeline(str(self.model_path), self.device)
        except Exception:
            self.device = "CPU"
            self.pipe = ov_genai.TextEmbeddingPipeline(str(self.model_path), "CPU")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document strings."""
        if not texts:
            return []
        # Intel NPU encoder plugin requires batch size 1
        results = []
        for t in texts:
            results.append(self.embed_query(t))
        return results

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        res = self.pipe.embed_documents([query])
        return res[0] if res else []
