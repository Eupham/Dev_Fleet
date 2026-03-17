import modal
import toml

# inference/utils.py
def start_logic(self, cfg: dict):
    from llama_cpp import Llama
    import os
    model_path = f"/root/models/{cfg['filename']}"
    
    # Add this debug check
    if not os.path.exists(model_path):
        print(f"CRITICAL: Model file not found at {model_path}")
        print(f"Contents of /root/models: {os.listdir('/root/models') if os.path.exists('/root/models') else 'DIR MISSING'}")
        
    self.llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=cfg["n_ctx"],
        verbose=False
    )

def get_tier_config(tier: str) -> dict:
    """Loads the specific model configuration from config.toml."""
    with open("inference/config.toml", "r") as f:
        config = toml.load(f)
    tier_cfg = config[tier]
    # Map 'model' to 'repo_id' to match function signature
    tier_cfg["repo_id"] = config["models"][tier]
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

def build_llama_image(repo_id: str, filename: str, **kwargs) -> modal.Image:
    """
    Compiles engine image. 
    **kwargs safely catches extra TOML fields (like gpu/timeout).
    """
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git")
        .run_commands("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
        .env({
            "HF_HOME": "/vol/cache",
            "CMAKE_ARGS": "-DGGML_CUDA=on",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64/stubs" 
        }) 
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0")
        .pip_install("llama-cpp-python", extra_options="--upgrade --no-cache-dir --force-reinstall")
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_function(download_weights, kwargs={"repo_id": repo_id, "filename": filename})
    )

class BaseInference:
    """Logic shared by all inference tiers. No __init__ to satisfy Modal."""
    
    def start_logic(self, cfg: dict):
        from llama_cpp import Llama
        # Use cfg.get('repo_id') which we mapped in get_tier_config
        repo = cfg.get("repo_id") or cfg.get("model")
        print(f"Loading {repo} from local SSD...")
        self.llm = Llama(
            model_path=f"/root/models/{cfg['filename']}",
            n_gpu_layers=-1, 
            n_ctx=cfg["n_ctx"],
            verbose=False
        )

    def generate_logic(self, messages, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema.model_json_schema()}}
        
        resp = self.llm.create_chat_completion(**kwargs)
        content = resp["choices"][0]["message"]["content"]
        
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except: return schema.model_construct()
        return content
