import modal
import toml
import os

def get_tier_config(tier: str) -> dict:
    """Loads the specific model configuration from config.toml."""
    with open("inference/config.toml", "r") as f:
        config = toml.load(f)
    tier_cfg = config[tier]
    tier_cfg["repo_id"] = config["models"][tier]
    return tier_cfg

def build_llama_image(repo_id: str, filename: str, **kwargs) -> modal.Image:
    """
    Compiles engine image and downloads weights directly via curl.
    """
    # Direct download URL for Hugging Face
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git", "curl")
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
        # Bypassing python library quirks to guarantee the raw binary is downloaded:
        .run_commands([
            "mkdir -p /root/models",
            f"curl -L -o /root/models/{filename} {download_url}"
        ])
    )

class BaseInference:
    """Logic shared by all inference tiers. No __init__ to satisfy Modal."""
    
    def start_logic(self, cfg: dict):
        from llama_cpp import Llama
        
        repo = cfg.get("repo_id") or cfg.get("model")
        model_path = f"/root/models/{cfg['filename']}"
        
        # --- DIAGNOSTIC CHECK ---
        print(f"Checking model path: {model_path}")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"File exists! Size: {size_mb:.2f} MB")
            if size_mb < 1.0:
                raise RuntimeError(f"CRITICAL: Downloaded a pointer file instead of the model! File size is only {size_mb} MB.")
        else:
            raise RuntimeError(f"CRITICAL: File NOT found at {model_path}. Build step failed to persist the file.")
        # ------------------------

        print(f"Loading {repo} from local SSD...")
        self.llm = Llama(
            model_path=model_path,
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
