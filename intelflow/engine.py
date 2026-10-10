"""
Core OpenVINO GenAI inference engine for Intel NPU.
Provides streaming generation, multi-turn KV-cache chat, and latency/throughput telemetry.
"""

import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Generator, Dict, Any, Optional, Callable, List

try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise RuntimeError(
        "OpenVINO GenAI runtime is required for LLM inference on Intel NPU.\n"
        "Please install it using:\n"
        "  pip install openvino-genai --break-system-packages\n"
        "or from AUR:\n"
        "  yay -S openvino-genai-bin"
    ) from e
from intelflow.config import BLOB_CACHE_DIR, resolve_model_tag
from intelflow.hub import ModelHub

class NPUEngine:
    def __init__(
        self,
        model_tag_or_path: str,
        device: str = "NPU",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the inference engine on the specified accelerator device.
        Defaults to 'NPU', with optional fallback to 'GPU' (Intel Arc) or 'CPU'.
        """
        self.device = device.upper()
        self.blob_cache_dir = cache_dir or BLOB_CACHE_DIR
        self.blob_cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve model directory
        if os.path.isdir(model_tag_or_path):
            self.model_path = Path(model_tag_or_path)
            self.model_tag = self.model_path.name
        else:
            self.model_tag = model_tag_or_path
            self.model_path = ModelHub.pull(model_tag_or_path)

        # Build device configuration with persistent graph cache
        ov_config: Dict[str, Any] = {
            "CACHE_DIR": str(self.blob_cache_dir),
            "PERFORMANCE_HINT": "LATENCY",
            "NPU_PLATFORM": "3720",
        }

        # Load GenAI Pipeline with graceful fallback
        try:
            self.pipe = ov_genai.LLMPipeline(str(self.model_path), self.device, ov_config)
        except Exception as e:
            if self.device == "NPU":
                # Fallback to GPU (Intel Arc) or CPU if NPU permissions/platform not yet available
                fallback_success = False
                for fallback_dev in ["GPU", "CPU"]:
                    try:
                        self.pipe = ov_genai.LLMPipeline(str(self.model_path), fallback_dev)
                        self.device = fallback_dev
                        fallback_success = True
                        break
                    except Exception:
                        pass
                if not fallback_success:
                    raise RuntimeError(
                        f"Failed to initialize OpenVINO pipeline on {device}: {e}. "
                        f"Run 'inpu doctor' to check driver readiness."
                    ) from e
            else:
                raise RuntimeError(
                    f"Failed to initialize OpenVINO pipeline on {self.device}: {e}."
                ) from e

        self.in_chat_session = False
        self.last_metrics: Dict[str, Any] = {}

    def start_chat(self, system_prompt: Optional[str] = None):
        """Start a multi-turn conversation with KV-cache persistence."""
        self.pipe.start_chat(system_prompt or "")
        self.in_chat_session = True

    def finish_chat(self):
        """End multi-turn conversation and flush KV-cache."""
        if self.in_chat_session:
            self.pipe.finish_chat()
            self.in_chat_session = False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Synchronous non-streaming generation.
        Returns response text and performance telemetry.
        """
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        config.do_sample = do_sample
        if do_sample:
            config.temperature = temperature
            config.top_p = top_p

        start_time = time.perf_counter()
        if self.in_chat_session:
            results = self.pipe.generate(prompt, config)
            text = str(results)
            pm = None
        else:
            results = self.pipe.generate([prompt], config)
            text = results.texts[0] if hasattr(results, "texts") and results.texts else str(results)
            pm = getattr(results, "perf_metrics", None)
        total_time = time.perf_counter() - start_time

        # Extract metrics
        metrics = {
            "total_time_sec": round(total_time, 3),
            "tokens_generated": 0,
            "throughput_tok_s": 0.0,
            "ttft_ms": 0.0,
        }
        if pm is not None:
            try:
                metrics["tokens_generated"] = pm.get_num_generated_tokens()
            except Exception:
                pass
            try:
                tp = pm.get_throughput()
                val = getattr(tp, "mean", tp)
                if float(val) > 0:
                    metrics["throughput_tok_s"] = round(float(val), 2)
            except Exception:
                pass
            try:
                ttft = pm.get_ttft()
                val = getattr(ttft, "mean", ttft)
                if float(val) > 0:
                    metrics["ttft_ms"] = round(float(val), 2)
            except Exception:
                pass

        if metrics["tokens_generated"] <= 0 and text:
            try:
                tok = self.pipe.get_tokenizer()
                metrics["tokens_generated"] = int(tok.encode(text).input_ids.shape[-1])
            except Exception:
                metrics["tokens_generated"] = max(1, int(len(text.split()) * 1.33))

        if metrics["throughput_tok_s"] <= 0 and total_time > 0 and metrics["tokens_generated"] > 0:
            metrics["throughput_tok_s"] = round(metrics["tokens_generated"] / total_time, 2)

        self.last_metrics = metrics
        return {"text": text, "metrics": metrics}

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Streaming token generator yielding delta string chunks in real time.
        """
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        config.do_sample = do_sample
        if do_sample:
            config.temperature = temperature
            config.top_p = top_p

        token_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0

        def streamer(chunk: str) -> bool:
            nonlocal first_token_time, token_count
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            token_queue.put(chunk)
            return False

        def worker():
            try:
                res = self.pipe.generate(prompt, config, streamer)
                token_queue.put(None)  # Sentinel to mark completion
            except Exception as ex:
                token_queue.put(ex)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = token_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        total_time = time.perf_counter() - start_time
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0.0
        tok_s = token_count / total_time if total_time > 0 else 0.0

        self.last_metrics = {
            "total_time_sec": round(total_time, 3),
            "tokens_generated": token_count,
            "throughput_tok_s": round(tok_s, 2),
            "ttft_ms": round(ttft, 2),
        }
