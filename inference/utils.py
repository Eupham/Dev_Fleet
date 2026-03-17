import modal
import toml

def get_tier_config(tier: str) -> dict:
    """Loads the specific model configuration from config.toml."""
    with open("inference/config.toml", "r") as f:
        config = toml.load(f)
    # Merge base settings with tier-specific settings
    tier_cfg = config[tier]
    tier_cfg["model"] = config["models"][tier]
    return tier_cfg

def download_weights(repo_id: str, filename: str):
    """Downloads weights directly into the Modal image build."""
    from huggingface_hub import hf_hub_download
    print(f"Downloading {repo_id}/{filename} into container image...")
    hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir="/root/models"
    )

def build_llama_image(repo_id: str, filename: str) -> modal.Image:
    """Compiles the engine image with latest dependencies and model weights."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git")
        # Link the CUDA stub so the compiler finds it on the CPU builder node
        .run_commands("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
        .env({
            "HF_HOME": "/vol/cache",
            "CMAKE_ARGS": "-DGGML_CUDA=on",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64/stubs" 
        }) 
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0")
        .pip_install(
            "llama-cpp-python", 
            extra_options="--upgrade --no-cache-dir --force-reinstall"
        )
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_function(download_weights, kwargs={"repo_id": repo_id, "filename": filename})
    )
