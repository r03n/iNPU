"""
OpenAI & Ollama Dual-Compatible REST API Server for iNPU.
Listens on port 52625 (FastFlowLM default) and serves LLM inference and embeddings
directly accelerated by Intel Core Ultra NPU (Meteor Lake).
"""

import time
import uuid
import json
from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from intelflow.engine import NPUEngine
from intelflow.embeddings import NPUEmbeddingEngine
from intelflow.hub import ModelHub
from intelflow.monitor import NPUTelemetry
from intelflow.config import DEFAULT_MODEL, resolve_model_tag

app = FastAPI(
    title="iNPU API",
    description="Dual OpenAI- and Ollama-compatible API server running on Intel Core Ultra NPU",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine state
current_engine: Optional[NPUEngine] = None
current_embed_engine: Optional[NPUEmbeddingEngine] = None
active_model_name: str = DEFAULT_MODEL
telemetry = NPUTelemetry()


def get_engine(model_name: Optional[str] = None) -> NPUEngine:
    global current_engine, active_model_name
    target_model = model_name or active_model_name or DEFAULT_MODEL

    if current_engine is None or current_engine.model_tag != target_model:
        current_engine = NPUEngine(target_model, device="NPU")
        active_model_name = target_model
    return current_engine


def get_embed_engine(model_name: Optional[str] = None) -> NPUEmbeddingEngine:
    global current_embed_engine
    if current_embed_engine is None:
        current_embed_engine = NPUEmbeddingEngine(model_name or "bge-small")
    return current_embed_engine


# Pydantic Schemas
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(default=512, alias="max_completion_tokens")
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class OllamaChatRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


class OllamaGenerateRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    prompt: str
    system: Optional[str] = ""
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


class EmbeddingRequest(BaseModel):
    model: Optional[str] = "bge-small"
    input: Union[str, List[str]]


class OllamaEmbeddingRequest(BaseModel):
    model: Optional[str] = "bge-small"
    prompt: Optional[str] = None
    input: Optional[Union[str, List[str]]] = None


class PullRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    insecure: Optional[bool] = False
    stream: Optional[bool] = False


# System & Version Endpoints
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": "NPU",
        "active_model": active_model_name,
        "engine_loaded": current_engine is not None,
    }


@app.get("/v1/version")
@app.get("/api/version")
def version():
    return {
        "version": "0.1.0",
        "backend": "openvino-npu",
        "arch": "3720",
        "device": "Intel(R) AI Boost",
    }


@app.get("/api/npu/status")
def npu_status_endpoint():
    metrics = telemetry.sample() if telemetry.is_npu_available() else {}
    return {
        "npu_available": telemetry.is_npu_available(),
        "active_model": active_model_name,
        "device": "Intel(R) AI Boost (NPU 3720)",
        "hardware_telemetry": metrics,
    }


# Models Endpoints (OpenAI & Ollama)
@app.get("/v1/models")
def list_models_openai():
    installed = ModelHub.list_installed()
    models_data = []
    now_ts = int(time.time())

    for m in installed:
        models_data.append({
            "id": m["tag"],
            "object": "model",
            "created": now_ts,
            "owned_by": "inpu",
        })

    if not any(m["id"] == active_model_name for m in models_data):
        models_data.append({
            "id": active_model_name,
            "object": "model",
            "created": now_ts,
            "owned_by": "inpu",
        })

    models_data.append({
        "id": "bge-small",
        "object": "model",
        "created": now_ts,
        "owned_by": "inpu",
    })

    return {"object": "list", "data": models_data}


@app.get("/api/tags")
def list_models_ollama():
    installed = ModelHub.list_installed()
    models = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for m in installed:
        models.append({
            "name": f"{m['tag']}:latest",
            "model": f"{m['tag']}:latest",
            "modified_at": now_iso,
            "size": int(m["size_mb"] * 1024 * 1024),
            "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_DNS, m['repo_id']).hex}",
            "details": {
                "parent_model": "",
                "format": "openvino",
                "family": m["tag"].split(":")[0],
                "parameter_size": "1.5B" if "1.5" in m["tag"] else "3B",
                "quantization_level": "INT4",
            },
        })

    if not models:
        models.append({
            "name": f"{active_model_name}:latest",
            "model": f"{active_model_name}:latest",
            "modified_at": now_iso,
            "size": 1500000000,
            "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_DNS, active_model_name).hex}",
            "details": {
                "format": "openvino",
                "family": "qwen",
                "parameter_size": "1.5B",
                "quantization_level": "INT4",
            },
        })

    return {"models": models}


@app.get("/api/ps")
def running_models_ollama():
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "models": [
            {
                "name": f"{active_model_name}:latest",
                "model": f"{active_model_name}:latest",
                "size": 1500000000,
                "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_DNS, active_model_name).hex}",
                "expires_at": "0001-01-01T00:00:00Z",
                "size_vram": 68722688,
                "details": {
                    "parent_model": "",
                    "format": "openvino",
                    "family": active_model_name.split(":")[0],
                    "parameter_size": "1.5B",
                    "quantization_level": "INT4",
                },
            }
        ]
    }


def format_messages(messages: List[ChatMessage]) -> str:
    """Format messages array into a prompt string for OpenVINO pipeline."""
    formatted = []
    for m in messages:
        if m.role == "system":
            formatted.append(f"<|im_start|>system\n{m.content}<|im_end|>")
        elif m.role == "user":
            formatted.append(f"<|im_start|>user\n{m.content}<|im_end|>")
        elif m.role == "assistant":
            formatted.append(f"<|im_start|>assistant\n{m.content}<|im_end|>")
    formatted.append("<|im_start|>assistant\n")
    return "\n".join(formatted)


# OpenAI Chat Completions
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    engine = get_engine(req.model)
    prompt = format_messages(req.messages)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.stream:
        async def event_generator():
            try:
                for chunk in engine.generate_stream(
                    prompt=prompt,
                    max_new_tokens=req.max_tokens or 512,
                    do_sample=(req.temperature or 0) > 0,
                    temperature=req.temperature or 0.7,
                    top_p=req.top_p or 0.9,
                ):
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                end_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(end_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_data = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err_data)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    res = engine.generate(
        prompt=prompt,
        max_new_tokens=req.max_tokens or 512,
        do_sample=(req.temperature or 0) > 0,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
    )

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": res["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": res["metrics"].get("tokens_generated", 0),
            "total_tokens": res["metrics"].get("tokens_generated", 0),
        },
        "metrics": res["metrics"],
    }


# OpenAI Legacy Text Completions
@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    engine = get_engine(req.model)
    raw_prompt = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.stream:
        async def event_generator():
            try:
                for chunk in engine.generate_stream(
                    prompt=raw_prompt,
                    max_new_tokens=req.max_tokens or 512,
                    do_sample=(req.temperature or 0) > 0,
                    temperature=req.temperature or 0.7,
                    top_p=req.top_p or 0.9,
                ):
                    data = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "text": chunk,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    res = engine.generate(
        prompt=raw_prompt,
        max_new_tokens=req.max_tokens or 512,
        do_sample=(req.temperature or 0) > 0,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
    )

    return {
        "id": req_id,
        "object": "text_completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "text": res["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": res["metrics"].get("tokens_generated", 0),
            "total_tokens": res["metrics"].get("tokens_generated", 0),
        },
    }


# Ollama Native Chat & Generate
@app.post("/api/chat")
async def ollama_chat(req: OllamaChatRequest):
    engine = get_engine(req.model)
    prompt = format_messages(req.messages)
    created_iso = datetime.now(timezone.utc).isoformat()
    max_tokens = (req.options or {}).get("num_predict", 512)
    temp = (req.options or {}).get("temperature", 0.7)
    top_p = (req.options or {}).get("top_p", 0.9)

    if req.stream:
        async def event_generator():
            token_count = 0
            start_t = time.perf_counter()
            for chunk in engine.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                do_sample=temp > 0,
                temperature=temp,
                top_p=top_p,
            ):
                token_count += 1
                data = {
                    "model": req.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                }
                yield f"{json.dumps(data)}\n"

            dur_ns = int((time.perf_counter() - start_t) * 1e9)
            final_data = {
                "model": req.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": dur_ns,
                "load_duration": 1000000,
                "prompt_eval_count": len(prompt.split()),
                "prompt_eval_duration": int(dur_ns * 0.1),
                "eval_count": token_count,
                "eval_duration": int(dur_ns * 0.9),
            }
            yield f"{json.dumps(final_data)}\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    start_t = time.perf_counter()
    res = engine.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        do_sample=temp > 0,
        temperature=temp,
        top_p=top_p,
    )
    dur_ns = int((time.perf_counter() - start_t) * 1e9)
    tokens = res["metrics"].get("tokens_generated", 1)

    return {
        "model": req.model,
        "created_at": created_iso,
        "message": {"role": "assistant", "content": res["text"]},
        "done": True,
        "total_duration": dur_ns,
        "eval_count": tokens,
        "eval_duration": dur_ns,
    }


@app.post("/api/generate")
async def ollama_generate(req: OllamaGenerateRequest):
    engine = get_engine(req.model)
    full_prompt = f"<|im_start|>system\n{req.system}<|im_end|>\n<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n" if req.system else req.prompt
    max_tokens = (req.options or {}).get("num_predict", 512)
    temp = (req.options or {}).get("temperature", 0.7)
    top_p = (req.options or {}).get("top_p", 0.9)

    if req.stream:
        async def event_generator():
            token_count = 0
            start_t = time.perf_counter()
            for chunk in engine.generate_stream(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                do_sample=temp > 0,
                temperature=temp,
                top_p=top_p,
            ):
                token_count += 1
                data = {
                    "model": req.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "response": chunk,
                    "done": False,
                }
                yield f"{json.dumps(data)}\n"

            dur_ns = int((time.perf_counter() - start_t) * 1e9)
            final_data = {
                "model": req.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": "",
                "done": True,
                "total_duration": dur_ns,
                "eval_count": token_count,
            }
            yield f"{json.dumps(final_data)}\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    start_t = time.perf_counter()
    res = engine.generate(
        prompt=full_prompt,
        max_new_tokens=max_tokens,
        do_sample=temp > 0,
        temperature=temp,
        top_p=top_p,
    )
    dur_ns = int((time.perf_counter() - start_t) * 1e9)
    tokens = res["metrics"].get("tokens_generated", 1)

    return {
        "model": req.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": res["text"],
        "done": True,
        "total_duration": dur_ns,
        "eval_count": tokens,
    }


# Embeddings Endpoints (OpenAI & Ollama)
@app.post("/v1/embeddings")
async def create_embeddings_openai(req: EmbeddingRequest):
    embed_engine = get_embed_engine(req.model)
    inputs = [req.input] if isinstance(req.input, str) else req.input
    vectors = embed_engine.embed_documents(inputs)

    data = []
    for i, vec in enumerate(vectors):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": vec,
        })

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": sum(len(x.split()) for x in inputs),
            "total_tokens": sum(len(x.split()) for x in inputs),
        }
    }


@app.post("/api/embeddings")
async def create_embeddings_ollama(req: OllamaEmbeddingRequest):
    embed_engine = get_embed_engine(req.model)
    text = req.prompt or (req.input if isinstance(req.input, str) else (req.input[0] if req.input else ""))
    vector = embed_engine.embed_query(text)
    return {"embedding": vector}


# Pull & Cancel Endpoints
@app.post("/api/pull")
def pull_endpoint(req: PullRequest, background_tasks: BackgroundTasks):
    target = req.name or req.model or DEFAULT_MODEL
    background_tasks.add_task(ModelHub.pull, target)
    return {"status": "pulling", "model": target}


@app.post("/api/cancel")
def cancel_endpoint(req: Dict[str, Any]):
    return {"status": "ok", "message": "Request cancelled"}
