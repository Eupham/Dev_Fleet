"""Microservice A — vLLM Inference Engine with GPU Memory Snapshots.

Hosts Qwen/Qwen2.5-Coder-32B-Instruct behind an OpenAI-compatible API on
a single A100-80 GB GPU.  Model weights are cached in Modal Volumes.
GPU memory snapshots eliminate cold-start JIT compilation overhead so we
never need to keep a GPU warm (scales to zero).

Deployment:
    modal deploy inference/server.py

Logs:
    modal app logs devfleet-inference
"""

import socket
import subprocess

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINUTES = 60  # seconds
VLLM_PORT = 8000
N_GPU = 1

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
SERVED_MODEL_NAME = "llm"  # short alias used by Microservice B

# ---------------------------------------------------------------------------
# Container image — vLLM + HuggingFace tooling
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Required for GPU snapshot compatibility
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

# Make `requests` importable inside the container for health-polling
with vllm_image.imports():
    import requests

# ---------------------------------------------------------------------------
# Volumes — persistent caches for weights and vLLM compilation artifacts
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("hf-cache-vol", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------

app = modal.App(
    "devfleet-inference",
    secrets=[modal.Secret.from_name("devfleet-modal-secrets")],
)

# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _wait_ready(proc: subprocess.Popen) -> None:
    """Busy-poll until the vLLM server is accepting connections."""
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {proc.returncode}")


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


def _sleep() -> None:
    """Offload model weights to CPU and empty the KV cache before snapshot."""
    requests.post(
        f"http://localhost:{VLLM_PORT}/sleep?level=1"
    ).raise_for_status()


def _wake_up() -> None:
    """Reload model weights onto GPU after snapshot restore."""
    requests.post(
        f"http://localhost:{VLLM_PORT}/wake_up"
    ).raise_for_status()


# ---------------------------------------------------------------------------
# vLLM Server class with GPU memory snapshots (scales to zero)
# ---------------------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
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
        """Start vLLM, warm up, then sleep before the snapshot is taken."""
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
            "--uvicorn-log-level=info",
            "--tensor-parallel-size",
            str(N_GPU),
            "--gpu-memory-utilization",
            "0.90",
            # Snapshot-specific flags
            "--enable-sleep-mode",
            "--max-num-seqs",
            "4",
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            "8192",
        ]

        print("[devfleet-inference] Starting vLLM:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)
        _wait_ready(self.proc)
        _warmup()
        _sleep()
        print("[devfleet-inference] Snapshot ready — server asleep.")

    @modal.enter(snap=False)
    def restore(self):
        """Wake the server back up after restoring from a GPU snapshot."""
        _wake_up()
        _wait_ready(self.proc)
        print("[devfleet-inference] Restored from snapshot — server live.")

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        """Expose the vLLM OpenAI-compatible API to the internet."""

    @modal.exit()
    def stop(self):
        self.proc.terminate()

