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

# --- NEW: Download function to run during image build ---
def download_weights(repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download
    print(f"Downloading {repo_id}/{filename} into container image...")
    hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir="/root/models"  # Store locally in the image, not a Volume!
    )

def build_llama_image(repo_id: str, filename: str) -> modal.Image:
    """Compiles the engine image with latest dependencies and bakes in model weights."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang")
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0")
        .pip_install("llama-cpp-python", extra_options="--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        # --- NEW: Bake the weights directly into the image ---
        .run_function(download_weights, kwargs={"repo_id": repo_id, "filename": filename})
    )
