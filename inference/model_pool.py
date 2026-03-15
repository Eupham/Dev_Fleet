"""Model Pool — Tier-based inference classes for difficulty routing.

Each tier is a separate Modal cls with its own image, GPU type, and vLLM instance.
The difficulty module maps task difficulty scores to tiers, and the llm_client
routes to the appropriate class via modal.Cls.from_name.

Tier → Model → Params → GPU:
  trivial  → Qwen3-4B                     → 4B dense  → T4
  simple   → Qwen3-8B                     → 8B dense  → T4
  moderate → Qwen3-Coder-30B-A3B-Instruct → 3B active → A10G  (primary)
  complex  → Qwen3-Coder-30B-A3B-Instruct → 3B active → A10G  (same as moderate)
  expert   → Qwen3-32B                    → 32B dense → A100-80GB  (~64GB BF16)

Note: moderate and complex share the same A10G instance (Qwen3-Coder-30B-A3B).
The primary Inference class in server.py serves the moderate/complex tier.
All models are ≤80B total parameters.
"""

from __future__ import annotations

from typing import Any, Optional
import modal
from fleet_app import app

MINUTES = 60
VLLM_PORT = 8001  # Offset to avoid collision with primary Inference on 8000

# ---------------------------------------------------------------------------
# Small tier — Qwen3-4B (T4 GPU, trivial tasks)
# ---------------------------------------------------------------------------

_SMALL_MODEL = "Qwen/Qwen3-4B"
_SMALL_SERVED = "llm-small"

_small_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .add_local_python_source("fleet_app", copy=True)
    .uv_pip_install("vllm==0.17.1", "hf_transfer")
    .run_commands(
        [f"huggingface-cli download {_SMALL_MODEL}"],
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    .env({
        "VLLM_SERVER_DEV_MODE": "1",
        "HF_HUB_OFFLINE": "1",
        "VLLM_LOGGING_LEVEL": "WARNING",
    })
)

with _small_image.imports():
    import requests as _requests_small


@app.cls(
    image=_small_image,
    gpu="T4",
    scaledown_window=2,
    timeout=10 * MINUTES,
    retries=0,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=50)
class InferenceSmall:
    """Qwen3-4B on T4 — trivial and simple tasks."""

    @modal.enter(snap=True)
    def start(self):
        import subprocess, socket, time
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [
            "vllm", "serve", _SMALL_MODEL,
            "--served-model-name", _SMALL_SERVED,
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--uvicorn-log-level=warning",
            "--enable-sleep-mode",
            "--dtype=bfloat16",
            "--max-num-seqs", "8",
            "--max-model-len", "8192",
        ]
        self.proc = subprocess.Popen(cmd)
        # Wait for ready
        while True:
            try:
                socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
                r = _requests_small.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vLLM small exited: {self.proc.returncode}")
                time.sleep(1)
        # Warmup then sleep for snapshot
        _requests_small.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json={"model": _SMALL_SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=120,
        )
        _requests_small.post(f"http://localhost:{VLLM_PORT}/sleep?level=1", timeout=60).raise_for_status()

    @modal.enter(snap=False)
    def restore(self):
        _requests_small.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=120).raise_for_status()

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        model: str = _SMALL_SERVED,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        schema: Optional[Any] = None,
    ) -> Any:
        import json as _json
        payload = {
            "model": model or _SMALL_SERVED,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        resp = _requests_small.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions", json=payload
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if schema:
            return schema.model_validate_json(content)
        return content

    @modal.exit()
    def stop(self):
        self.proc.terminate()


# ---------------------------------------------------------------------------
# Medium tier — Qwen3-8B (T4 GPU, simple tasks)
# ---------------------------------------------------------------------------

_MEDIUM_MODEL = "Qwen/Qwen3-8B"
_MEDIUM_SERVED = "llm-medium"
_MEDIUM_PORT = 8002

_medium_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .add_local_python_source("fleet_app", copy=True)
    .uv_pip_install("vllm==0.17.1", "hf_transfer")
    .run_commands(
        [f"huggingface-cli download {_MEDIUM_MODEL}"],
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    .env({
        "VLLM_SERVER_DEV_MODE": "1",
        "HF_HUB_OFFLINE": "1",
        "VLLM_LOGGING_LEVEL": "WARNING",
    })
)

with _medium_image.imports():
    import requests as _requests_medium


@app.cls(
    image=_medium_image,
    gpu="T4",
    scaledown_window=2,
    timeout=10 * MINUTES,
    retries=0,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=30)
class InferenceMedium:
    """Qwen3-8B on T4 — simple tasks."""

    @modal.enter(snap=True)
    def start(self):
        import subprocess, socket, time
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [
            "vllm", "serve", _MEDIUM_MODEL,
            "--served-model-name", _MEDIUM_SERVED,
            "--host", "0.0.0.0", "--port", str(_MEDIUM_PORT),
            "--uvicorn-log-level=warning",
            "--enable-sleep-mode",
            "--dtype=bfloat16",
            "--max-num-seqs", "4",
            "--max-model-len", "8192",
        ]
        self.proc = subprocess.Popen(cmd)
        while True:
            try:
                socket.create_connection(("localhost", _MEDIUM_PORT), timeout=1).close()
                r = _requests_medium.get(f"http://localhost:{_MEDIUM_PORT}/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vLLM medium exited: {self.proc.returncode}")
                time.sleep(1)
        _requests_medium.post(
            f"http://localhost:{_MEDIUM_PORT}/v1/chat/completions",
            json={"model": _MEDIUM_SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=120,
        )
        _requests_medium.post(f"http://localhost:{_MEDIUM_PORT}/sleep?level=1", timeout=60).raise_for_status()

    @modal.enter(snap=False)
    def restore(self):
        _requests_medium.post(f"http://localhost:{_MEDIUM_PORT}/wake_up", timeout=120).raise_for_status()

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        model: str = _MEDIUM_SERVED,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        schema: Optional[Any] = None,
    ) -> Any:
        payload = {
            "model": model or _MEDIUM_SERVED,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        resp = _requests_medium.post(
            f"http://localhost:{_MEDIUM_PORT}/v1/chat/completions", json=payload
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if schema:
            return schema.model_validate_json(content)
        return content

    @modal.exit()
    def stop(self):
        self.proc.terminate()


# ---------------------------------------------------------------------------
# Large tier — Qwen3-32B (A100-80GB, expert tasks)
# 32B dense model fits on a single A100-80GB in BF16 (~64GB VRAM).
# ---------------------------------------------------------------------------

_LARGE_MODEL = "Qwen/Qwen3-32B"
_LARGE_SERVED = "llm-large"
_LARGE_PORT = 8003

_large_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .add_local_python_source("fleet_app", copy=True)
    .uv_pip_install("vllm==0.17.1", "hf_transfer")
    .run_commands(
        [f"huggingface-cli download {_LARGE_MODEL}"],
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    .env({
        "VLLM_SERVER_DEV_MODE": "1",
        "HF_HUB_OFFLINE": "1",
        "VLLM_LOGGING_LEVEL": "WARNING",
    })
)

with _large_image.imports():
    import requests as _requests_large


@app.cls(
    image=_large_image,
    gpu="A100-80GB",
    scaledown_window=2,
    timeout=20 * MINUTES,
    retries=0,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=10)
class InferenceLarge:
    """Qwen3-32B on A100-80GB — expert tasks (32B dense, ~64GB BF16)."""

    @modal.enter(snap=True)
    def start(self):
        import subprocess, socket, time
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [
            "vllm", "serve", _LARGE_MODEL,
            "--served-model-name", _LARGE_SERVED,
            "--host", "0.0.0.0", "--port", str(_LARGE_PORT),
            "--uvicorn-log-level=warning",
            "--enable-sleep-mode",
            "--dtype=bfloat16",
            "--max-num-seqs", "2",
            "--max-model-len", "8192",
        ]
        self.proc = subprocess.Popen(cmd)
        while True:
            try:
                socket.create_connection(("localhost", _LARGE_PORT), timeout=1).close()
                r = _requests_large.get(f"http://localhost:{_LARGE_PORT}/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vLLM large exited: {self.proc.returncode}")
                time.sleep(1)
        _requests_large.post(
            f"http://localhost:{_LARGE_PORT}/v1/chat/completions",
            json={"model": _LARGE_SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=300,
        )
        _requests_large.post(f"http://localhost:{_LARGE_PORT}/sleep?level=1", timeout=120).raise_for_status()

    @modal.enter(snap=False)
    def restore(self):
        _requests_large.post(f"http://localhost:{_LARGE_PORT}/wake_up", timeout=180).raise_for_status()

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        model: str = _LARGE_SERVED,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        schema: Optional[Any] = None,
    ) -> Any:
        payload = {
            "model": model or _LARGE_SERVED,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        resp = _requests_large.post(
            f"http://localhost:{_LARGE_PORT}/v1/chat/completions", json=payload
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if schema:
            return schema.model_validate_json(content)
        return content

    @modal.exit()
    def stop(self):
        self.proc.terminate()
