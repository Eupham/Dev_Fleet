import modal
import tomllib
from pathlib import Path

_possible_paths = [
    Path(__file__).parent / "vllm_config.toml",
    Path("/root/inference/vllm_config.toml"),
    Path("inference/vllm_config.toml")
]
config = {}
for p in _possible_paths:
    if p.exists():
        with open(p, "rb") as f: config = tomllib.load(f)
        break

def get_tier_config(tier: str) -> dict:
    c = config[tier].copy()
    c["model"] = config["models"][tier]
    return c

def build_llama_image(repo_id: str, filename: str) -> modal.Image:
    # Notice we are injecting langgraph and mcp here for the orchestrator
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .pip_install("huggingface_hub", "hf_transfer", "langgraph", "mcp")
        .env({"CMAKE_ARGS": "-DGGML_CUDA=on", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .pip_install("llama-cpp-python")
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/vllm_config.toml", remote_path="/root/inference/vllm_config.toml", copy=True)
        .run_commands([
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{repo_id}', allow_patterns=['{filename}*'])\""
        ])
    )
