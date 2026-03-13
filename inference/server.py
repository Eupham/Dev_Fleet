"""Inference Engine — vLLM with GPU Memory Snapshots.

Hosts Qwen/Qwen2.5-Coder-0.5B-Instruct behind an OpenAI-compatible API on
a single A100-80 GB GPU.  Model weights are cached in Modal Volumes.
GPU memory snapshots eliminate cold-start JIT compilation overhead so we
never need to keep a GPU warm (scales to zero).

The ``Inference`` class exposes two interfaces:
  * ``@modal.web_server`` — external OpenAI-compatible HTTP endpoint.
  * ``@modal.method`` (``generate``) — Modal-native RPC used by the
    orchestrator, avoiding HTTP overhead and idle-timeout waste.
"""

import socket
import subprocess
import time
from typing import Any, Optional

import modal

from fleet_app import app  # shared app defined in app.py

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINUTES = 60  # seconds
VLLM_PORT = 8000
N_GPU = 1

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SERVED_MODEL_NAME = "llm"  # short alias used by the orchestrator

# ---------------------------------------------------------------------------
# Container image — vLLM + HuggingFace tooling
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .uv_pip_install(
        "vllm==0.7.3",
        "huggingface-hub==0.36.0",
        "hf_transfer",
        "outlines>=0.0.38",
    )
    .run_commands(
        [
            f"huggingface-cli download {MODEL_NAME}",
        ]
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_SERVER_DEV_MODE": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # Required for snapshot survival without NCCL socket crashes
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

# Make `requests` importable inside the container for health-polling
with vllm_image.imports():
    import requests
    import json
    import outlines

# ---------------------------------------------------------------------------
# Volumes — persistent caches for weights and vLLM compilation artifacts
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("hf-cache-vol", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _wait_ready(proc: subprocess.Popen) -> None:
    """Busy-poll until the vLLM server is accepting connections."""
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            # Double check with an HTTP health endpoint
            resp = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
            if resp.status_code == 200:
                return
        except (OSError, requests.exceptions.RequestException):
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {proc.returncode}")
            time.sleep(1)


def _warmup() -> None:
    """Run a few inference requests to capture CUDA graphs in the snapshot."""
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        ).raise_for_status()

# ---------------------------------------------------------------------------
# vLLM Server class with GPU memory snapshots (scales to zero)
# ---------------------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu="T4",
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    retries=0,
    volumes={
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class Inference:
    """OpenAI-compatible vLLM server with GPU snapshot support."""

    @modal.enter(snap=True)
    def start(self):
        """Start vLLM, warm up, and snapshot."""
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=warning",
            "--tensor-parallel-size",
            str(N_GPU),
            "--gpu-memory-utilization",
            "0.90",
            "--disable-log-requests",  # Silences the massive prompt strings in logs
            # Snapshot-specific flags
            "--enforce-eager",
            "--dtype=half",
            "--max-num-seqs",
            "4",
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            "8192",
        ]

        print("[dev_fleet] Starting vLLM:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)
        _wait_ready(self.proc)
        _warmup()
        print("[dev_fleet] Snapshot ready — capturing full GPU memory including weights.")

    @modal.enter(snap=False)
    def restore(self):
        """Resume server live after restoring from a GPU snapshot."""
        _wait_ready(self.proc)
        print("[dev_fleet] Restored from snapshot — server live.")

    @modal.method()
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "llm",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        schema: Optional[Any] = None,
    ) -> Any:
        """Modal-native RPC — called by the orchestrator via ``.remote()``.

        Forwards the request to the local vLLM subprocess and returns
        the generated text directly.  No external HTTP round-trip.
        If a Pydantic schema is provided, uses vLLM's native JSON mode to constrain the output.
        """
        if schema:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema()
                    }
                }
            }
            resp = requests.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Validate and return the Pydantic model directly
            return schema.model_validate_json(content)

        else:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = requests.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        """Expose the vLLM OpenAI-compatible API to the internet."""

    @modal.exit()
    def stop(self):
        self.proc.terminate()
