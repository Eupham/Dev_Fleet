"""Inference Engine — vLLM with GPU Memory Snapshots.

Hosts Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 behind an OpenAI-compatible API on
a single L40S GPU (48 GB VRAM).  Model weights are cached in Modal Volumes.
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

import atexit
import os
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

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
SERVED_MODEL_NAME = "llm"  # short alias for all orchestrator calls

# ---------------------------------------------------------------------------
# VRAM Registry
# ---------------------------------------------------------------------------
# Schema: (total_params_B, bytes_per_param, quant_label, vision_overhead_gb)
#
# bytes_per_param:
#   bf16 / fp16          → 2.0
#   fp8                  → 1.0
#   int8                 → 1.0
#   gptq-int4 / moe_wna16 / awq-int4  → 0.5
#
# vision_overhead_gb:
#   Text-only models (Qwen3, QwQ, Qwen2.5-Coder)  → 0.0
#   Multimodal models (Qwen3.5)                    → 8.0  (conservative estimate)
#   Pass language_model_only=True to zero this out.
#
# MoE models: always use TOTAL parameter count, not active parameter count.
# All expert weights must reside in VRAM regardless of activation sparsity.
MODEL_REGISTRY: dict[str, tuple[float, float, str, float]] = {
    # ── Qwen3.5 multimodal MoE (GDN+MoE, requires vLLM nightly) ────────────
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": (35.0, 0.5, "moe_wna16", 8.0),
    "Qwen/Qwen3.5-35B-A3B-FP8":       (35.0, 1.0, "fp8",        8.0),
    "Qwen/Qwen3.5-35B-A3B":           (35.0, 2.0, "bf16",       8.0),
    "Qwen/Qwen3.5-27B":               (27.0, 2.0, "bf16",       8.0),
    "Qwen/Qwen3.5-9B":                (9.0,  2.0, "bf16",       8.0),
    # ── Qwen3 text-only (stable vLLM, AWQ available) ────────────────────────
    "Qwen/Qwen3-8B-Instruct":               (8.0,  2.0, "bf16",     0.0),
    "Qwen/Qwen3-8B-Instruct-AWQ":           (8.0,  0.5, "awq-int4", 0.0),
    "Qwen/Qwen3-32B-Instruct":              (32.0, 2.0, "bf16",     0.0),
    "Qwen/Qwen3-32B-Instruct-AWQ":          (32.0, 0.5, "awq-int4", 0.0),
    "Qwen/Qwen3-30B-A3B-Instruct":          (30.0, 2.0, "bf16",     0.0),
    "Qwen/Qwen3-30B-A3B-Instruct-AWQ":      (30.0, 0.5, "awq-int4", 0.0),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct":    (30.0, 2.0, "bf16",     0.0),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ":(30.0, 0.5, "awq-int4", 0.0),
}

_VRAM_HEADROOM = 0.75  # 25% reserved for KV cache, CUDA graphs, fragmentation


def _get_gpu_vram_gb() -> float:
    """Return total VRAM in GB for GPU 0."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return float(raw) / 1024  # MiB → GB


def assert_model_fits(model_id: str, language_model_only: bool = False) -> None:
    """
    Raise before launching vLLM if the model cannot fit in available VRAM.
    required_gb = (params_B * bytes_per_param) + vision_overhead_gb
    budget_gb   = gpu_vram_gb * 0.75
    PASS iff required_gb <= budget_gb
    """
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Model {model_id!r} is not in MODEL_REGISTRY.\n"
            "Add (total_params_B, bytes_per_param, quant_label, vision_overhead_gb) "
            "before deploying. Look up the model card — do not guess."
        )
    params_b, bytes_pp, quant_label, vision_gb = MODEL_REGISTRY[model_id]
    effective_vision = 0.0 if language_model_only else vision_gb
    required_gb = (params_b * bytes_pp) + effective_vision
    gpu_vram_gb = _get_gpu_vram_gb()
    budget_gb = gpu_vram_gb * _VRAM_HEADROOM
    if required_gb > budget_gb:
        vision_note = "  [vision skipped via --language-model-only]" if language_model_only else ""
        raise ValueError(
            f"\nVRAM preflight FAILED — do not launch vLLM.\n"
            f"  model      : {model_id} ({quant_label})\n"
            f"  weights    : {params_b * bytes_pp:.1f} GB  "
            f"({params_b}B params × {bytes_pp} B/param)\n"
            f"  vision     : {effective_vision:.1f} GB{vision_note}\n"
            f"  total need : {required_gb:.1f} GB\n"
            f"  GPU budget : {budget_gb:.1f} GB  "
            f"({gpu_vram_gb:.1f} GB × {_VRAM_HEADROOM} headroom)\n"
            f"  shortfall  : {required_gb - budget_gb:.1f} GB\n\n"
            "Fix options:\n"
            "  1. Pass --language-model-only if vision is not needed (saves ~8 GB)\n"
            "  2. Use a quantized variant (GPTQ-Int4 or FP8)\n"
            "  3. Provision a larger GPU — L40S minimum for this model\n"
            "  4. Use Qwen/Qwen3-8B-Instruct on A10 for smoketests"
        )


def assert_vllm_supports_architecture(model_id: str) -> None:
    """
    For Qwen3.5 models, confirm the installed vLLM has the GDN architecture
    registered. Stable releases prior to Qwen3.5 support will fail with a
    ValidationError that looks identical to other startup failures.
    """
    if "Qwen3.5" not in model_id:
        return
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError(
            "vLLM is not installed. "
            "Run: uv pip install vllm --torch-backend=auto "
            "--extra-index-url https://wheels.vllm.ai/nightly"
        )
    try:
        from vllm.model_executor.models import ModelRegistry
        required = "Qwen3_5MoeForConditionalGeneration"
        if required not in ModelRegistry.get_supported_archs():
            raise RuntimeError(
                f"Installed vLLM does not support {required}.\n"
                "Install the correct version:\n"
                "  uv pip install vllm --torch-backend=auto "
                "--extra-index-url https://wheels.vllm.ai/nightly\n"
                "Pin the resulting version after confirming it works."
            )
    except ImportError:
        # vLLM is installed but its CUDA extension (vllm._C) cannot be loaded
        # via direct Python import in this context (libtorch_cuda.so path not
        # set up). This is normal — CUDA paths are configured inside the vllm
        # serve subprocess. Skip the arch check; vllm serve will validate it.
        pass

# ---------------------------------------------------------------------------
# Container image — vLLM nightly + HuggingFace tooling
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    # --torch-backend=cu129: explicit CUDA 12.9 backend matching the base image.
    # "auto" fails during Modal image build because no GPU driver is present,
    # causing uv to install CPU-only PyTorch (missing libtorch_cuda.so).
    .uv_pip_install(
        "vllm",
        "hf_transfer",
        "transformers>=4.53.0",
        extra_options="--torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/nightly",
    )
    .run_commands(
        [
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_NAME}')\"",
            f"HF_HUB_ENABLE_HF_TRANSFER=1 python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_NAME}', allow_patterns=['*.json', '*.bin', '*.safetensors', '*.model'])\"",
        ],
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"},
    )
    .env(
        {
            # Enable /sleep and /wake_up endpoints for GPU snapshot support.
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # Keep vLLM logs at WARNING to reduce modal app logs noise
            "VLLM_LOGGING_LEVEL": "WARNING",
            # Model is baked into the image at build time; skip hub network calls at runtime
            "HF_HUB_OFFLINE": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "0",
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
    """
    Poll until vLLM accepts connections or the process exits.
    A crashed process is NOT retried. Connection refused while the process
    is alive is treated as normal startup delay.

    Timeout is configurable via VLLM_READY_TIMEOUT environment variable (seconds).
    Default is 600s to allow for model loading + AOT compilation on large models.
    """
    timeout_s = int(os.environ.get("VLLM_READY_TIMEOUT", "600"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode} during startup.\n"
                "Do not retry with identical arguments — this failure is deterministic.\n"
                "Check stdout/stderr above for the root cause."
            )
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return  # server is ready
        except ConnectionRefusedError:
            time.sleep(1)
    proc.terminate()
    raise TimeoutError(
        f"vLLM did not become ready within {timeout_s}s. Process terminated."
    )


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
    """Build the vLLM serve command for Qwen3.5-35B-A3B-GPTQ-Int4 on L40S."""
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
        "0.85",
        "--quantization",
        "moe_wna16",           
        "--language-model-only",   
        "--reasoning-parser",
        "qwen3",               
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-sleep-mode",
        "--dtype=bfloat16",    
        "--max-num-seqs",
        "4",
        "--max-model-len",
        "131072",              
        "--max-num-batched-tokens",
        "131072",
    ]
# ---------------------------------------------------------------------------
# vLLM Server class with GPU memory snapshots (scales to zero)
# ---------------------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu="L40S",
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

        # Preflight checks — raise before launching vLLM if the model cannot fit.
        assert_vllm_supports_architecture(MODEL_NAME)
        assert_model_fits(MODEL_NAME, language_model_only=True)

        cmd = _build_serve_cmd()
        # Log only after the full command is constructed (Fix 4).
        print("[dev_fleet] Starting vLLM for snapshot:", " ".join(cmd))

        # Build environment for vLLM subprocess
        env = _os.environ.copy()
        # vLLM environment handled via image .env block
        # Keep existing vLLM env vars from image
        env.setdefault("VLLM_SERVER_DEV_MODE", "1")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
        env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "0")
        env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
        env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

        self.proc = subprocess.Popen(cmd, env=env)
        # Register the plain helper (not the @modal.exit method) to avoid
        # KeyError when Modal's _partial_function.__get__ is called during
        # atexit before the cls machinery is fully initialised.
        atexit.register(self._terminate_proc)
        try:
            _wait_ready(self.proc)
        except Exception:
            self._terminate_proc()
            raise
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
        # Ensure timeout is available for _wait_ready
        _os.environ.setdefault("VLLM_READY_TIMEOUT", "600")

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
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }

        resp = requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if schema:
            return schema.model_validate_json(content)
        return content

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        """Expose the vLLM OpenAI-compatible API to the internet."""

    def _terminate_proc(self):
        """Terminate the vLLM subprocess. Plain helper — safe to call from atexit."""
        if getattr(self, "proc", None) is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    @modal.exit()
    def stop(self):
        """Terminate vLLM on Modal container exit."""
        self._terminate_proc()

    def __exit__(self, *_):
        self._terminate_proc()
