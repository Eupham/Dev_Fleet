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

def download_weights(repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download
    print(f"Downloading {repo_id}/{filename} into container image...")
    hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir="/root/models",
        local_dir_use_symlinks=False, # Put this back to strictly enforce physical files
        force_download=True           # CRITICAL: Ignores broken cache and forces a fresh download
    )

def build_llama_image(repo_id: str, filename: str) -> modal.Image:
    """Compiles the engine image with latest dependencies and model weights."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        # 1. ADD cmake so we can compile the C++ backend
        .apt_install("build-essential", "clang", "cmake") 
        .env({
            "HF_HOME": "/vol/cache",
            # 2. ADD the compiler flag to force CUDA support
            "CMAKE_ARGS": "-DGGML_CUDA=on" 
        }) 
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0")
        # 3. REPLACE the wheel link with a forced source compilation
        .pip_install(
            "llama-cpp-python", 
            extra_options="--upgrade --no-cache-dir --force-reinstall"
        )
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_function(download_weights, kwargs={"repo_id": repo_id, "filename": filename})
    )
