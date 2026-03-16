import socket
import subprocess
import time
import tomllib
import os
from pathlib import Path
import modal

# Safely resolve TOML path whether local or executing inside the Modal container
_possible_paths = [
    Path(__file__).parent / "vllm_config.toml",
    Path("/root/inference/vllm_config.toml"),
    Path("inference/vllm_config.toml")
]
config = {}
for p in _possible_paths:
    if p.exists():
        with open(p, "rb") as f:
            config = tomllib.load(f)
        break

def get_tier_config(tier: str) -> dict:
    c = config[tier].copy()
    c["model"] = config["models"][tier]
    return c

def build_serve_cmd(cfg: dict) -> list[str]:
    cmd = [
        "vllm", "serve", cfg["model"],
        "--served-model-name", cfg["served_model_name"],
        "--host", "0.0.0.0",
        "--port", str(cfg["port"]),
        "--uvicorn-log-level=warning",
        "--enable-sleep-mode",
    ]
    if "tensor_parallel_size" in cfg: cmd.extend(["--tensor-parallel-size", str(cfg["tensor_parallel_size"])])
    if "gpu_memory_utilization" in cfg: cmd.extend(["--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])])
    if "quantization" in cfg: cmd.extend(["--quantization", cfg["quantization"]])
    if cfg.get("language_model_only"): cmd.append("--language-model-only")
    if "reasoning_parser" in cfg: cmd.extend(["--reasoning-parser", cfg["reasoning_parser"]])
    if cfg.get("enable_auto_tool_choice"): cmd.append("--enable-auto-tool-choice")
    if "tool_call_parser" in cfg: cmd.extend(["--tool-call-parser", cfg["tool_call_parser"]])
    if "dtype" in cfg: cmd.extend(["--dtype", cfg["dtype"]])
    if "max_num_seqs" in cfg: cmd.extend(["--max-num-seqs", str(cfg["max_num_seqs"])])
    if "max_model_len" in cfg: cmd.extend(["--max-model-len", str(cfg["max_model_len"])])
    if "max_num_batched_tokens" in cfg: cmd.extend(["--max-num-batched-tokens", str(cfg["max_num_batched_tokens"])])
    return cmd

def build_vllm_image(model_id: str, is_nightly: bool = False) -> modal.Image:
    img = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/vllm_config.toml", remote_path="/root/inference/vllm_config.toml")
    )
    if is_nightly:
        img = img.add_local_python_source("orchestrator", copy=True).uv_pip_install(
            "vllm", "hf_transfer", "transformers>=4.53.0",
            extra_options="--torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/nightly",
        )
    else:
        img = img.uv_pip_install("vllm==0.17.1", "hf_transfer")

    return (
        img.run_commands(
            [
                f'python -c "from huggingface_hub import snapshot_download; snapshot_download('{model_id}')"',
                f'HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('{model_id}', allow_patterns=['*.json', '*.bin', '*.safetensors', '*.model'])"',
            ],
            env={"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"},
        )
        .env({
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "HF_HUB_OFFLINE": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
            "TORCH_FR_BUFFER_SIZE ": "0",
            "PYTHONWARNINGS": "ignore::FutureWarning",
        })
    )

def wait_for_vllm(proc: subprocess.Popen, port: int, timeout_s: int = 600) -> None:
    import requests
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode} during startup.")
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise TimeoutError(f"vLLM did not become ready on port {port} within {timeout_s}s.")
