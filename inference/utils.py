import modal
import tomllib
from pathlib import Path

_possible_paths = [
    Path(__file__).parent / "config.toml",
    Path("/root/inference/config.toml"),
    Path("inference/config.toml")
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
    """Compiles the engine image with latest dependencies and model weights."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang")
        .pip_install("huggingface_hub", "hf_transfer", "langgraph>=1.1.2", "mcp>=1.26.0")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .pip_install("llama-cpp-python", extra_options="--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_commands([
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{repo_id}', allow_patterns=['{filename}'])\""
        ])
    )
