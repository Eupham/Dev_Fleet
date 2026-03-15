"""Inference Engine — vLLM with GPU Memory Snapshots.

Hosts Qwen/Qwen3-Coder-30B-A3B-Instruct behind an OpenAI-compatible API on
a single A10G GPU.  Model weights are cached in Modal Volumes.
GPU memory snapshots eliminate cold-start JIT compilation overhead so we
never need to keep a GPU warm (scales to zero).

The ``Inference`` class exposes two interfaces:
  * ``@modal.web_server`` — external OpenAI-compatible HTTP endpoint.
  * ``@modal.method`` (``generate``) — Modal-native RPC used by the
    orchestrator, avoiding HTTP overhead and idle-timeout waste.

Snapshot lifecycle (per Modal GPU snapshot docs):
  snap=True  — start vLLM, warm it, call /sleep to offload weights to CPU
               so the snapshot captures the sleeping server with weights in
               CPU memory (preserves compiled CUDA graphs).
  snap=False — call /wake_up to reload weights back to GPU, then wait ready.
               No JIT recompilation because CUDA graphs survived the snapshot.
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

MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
SERVED_MODEL_NAME = "llm"  # short alias for all orchestrator calls

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
        "vllm==0.17.1",
        "hf_transfer",
    )
    .run_commands(
        [
            f"huggingface-cli download {MODEL_NAME}",
        ],
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"},
    )
    .env(
        {
            # Enable /sleep and /wake_up endpoints for GPU snapshot support.
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # Required for snapshot survival without NCCL socket crashes
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # Prevent NCCL from probing hardware topology across physical nodes
            "NCCL_WARN_DISABLE": "1",
            # Keep vLLM logs at WARNING to reduce modal app logs noise
            "VLLM_LOGGING_LEVEL": "WARNING",
            # Model is baked into the image at build time; skip hub network calls at runtime
            "HF_HUB_OFFLINE": "1",
        }
    )
)

# Make `requests` importable inside the container for health-polling
with vllm_image.imports():
    import requests
    import json

# ---------------------------------------------------------------------------
# Volumes — persistent caches for vLLM compilation artifacts
# ---------------------------------------------------------------------------

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
    """Run a few inference requests to capture CUDA graphs before snapshot."""
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


def _build_serve_cmd() -> list[str]:
    """Build the vLLM serve command."""
    return [
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
        # Disable CUDA-IPC custom allocator — shared-memory handles don't survive
        # CRIU migration to a different physical node, causing CudaCheckpointException.
        "--disable-custom-all-reduce",
        # Enable sleep/wake_up endpoints (requires VLLM_SERVER_DEV_MODE=1)
        "--enable-sleep-mode",
        "--dtype=bfloat16",  # Ampere+ GPUs support bfloat16 natively
        "--max-num-seqs",
        "4",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "8192",
        # MoE routing is handled automatically by vLLM 0.17.1 — no extra flags needed.
    ]

# ---------------------------------------------------------------------------
# vLLM Server class with GPU memory snapshots (scales to zero)
# ---------------------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu="A10G",
    scaledown_window=2,  # Modal minimum (>0 required); snapshots handle cold-start
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
    """OpenAI-compatible vLLM server with GPU snapshot support.

    Snapshot lifecycle:
      snap=True  — start vLLM, warm it, call /sleep to put weights in CPU
                   memory before the snapshot is taken.
      snap=False — call /wake_up to reload weights from CPU to GPU.
                   CUDA graphs are preserved; no JIT recompilation needed.
    """

    @modal.enter(snap=True)
    def start(self):
        """Start vLLM, warm it, then put it to sleep for the snapshot."""
        import os as _os
        # Blind PyTorch/NCCL to any ghost GPUs visible to cgroups on multi-GPU chassis.
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        cmd = _build_serve_cmd()
        print("[dev_fleet] Starting vLLM for snapshot:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)
        _wait_ready(self.proc)
        _warmup()

        # Put vLLM to sleep: offload weights to CPU memory so the snapshot
        # captures them. CUDA graphs are preserved across the sleep boundary.
        print("[dev_fleet] Putting vLLM to sleep (level=1) for snapshot capture...")
        resp = requests.post(
            f"http://localhost:{VLLM_PORT}/sleep?level=1",
            timeout=120,
        )
        resp.raise_for_status()
        print("[dev_fleet] vLLM sleeping — snapshot will capture CPU-side weights + CUDA graphs.")

    @modal.enter(snap=False)
    def restore(self):
        """Wake up the sleeping vLLM server after container thaws from snapshot."""
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # The snapshot contains a sleeping vLLM subprocess. Wake it up —
        # weights reload from CPU to GPU, CUDA graphs resume. No JIT needed.
        print("[dev_fleet] Waking vLLM from snapshot...")
        resp = requests.post(
            f"http://localhost:{VLLM_PORT}/wake_up",
            timeout=120,
        )
        resp.raise_for_status()
        _wait_ready(self.proc)
        print("[dev_fleet] Restored from snapshot — server live, no JIT compilation.")

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
                        "schema": schema.model_json_schema(),
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
