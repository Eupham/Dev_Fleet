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
    Compiles engine image and downloads weights.
    Fixes the 'unknown model architecture: qwen35' error by forcing a 
    submodule update to the latest GDN-enabled C++ engine.
    """
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git", "curl", "ninja-build")
        # Pre-link the CUDA stubs for the build step
        .run_commands("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
        .env({"HF_HOME": "/vol/cache"}) 
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0")
        .run_commands([
            # 1. Clone the Python wrapper
            "git clone --depth 1 --recurse-submodules https://github.com/abetlen/llama-cpp-python.git /tmp/llama-cpp-python",
            
            # 2. Update the internal engine to the latest master (enables 'qwen35' GDN support)
            "cd /tmp/llama-cpp-python/vendor/llama.cpp && git fetch origin master && git checkout origin/master",
            
            # 3. SURGICAL FIX: Delete all references to the broken 'mtmd' tool in the build files.
            # This is the only way to bypass the 'incorrect number of arguments' error 
            # while keeping the rest of the new Qwen 3.5 code.
            "find /tmp/llama-cpp-python -name 'CMakeLists.txt' -exec sed -i '/mtmd/d' {} +",
            
            # 4. Compile with CUDA and Ninja
            "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs && "
            "cd /tmp/llama-cpp-python && "
            "CMAKE_ARGS='-DGGML_CUDA=on -G Ninja -DLLAMA_BUILD_TOOLS=OFF' pip install ."
        ])
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_commands([
            "mkdir -p /root/models",
            f"curl -L -o /root/models/{filename} {download_url}"
        ])
    )

class BaseInference:
    """Logic shared by all inference tiers."""
    
    def start_logic(self, cfg: dict):
        from llama_cpp import Llama
        repo = cfg.get("repo_id") or cfg.get("model")
        model_path = f"/root/models/{cfg['filename']}"
        
        print(f"Loading {repo} with Qwen 3.5 GDN support...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=cfg["n_ctx"],
            verbose=True,    # Verify the 'qwen35' architecture loads in logs
            use_mmap=True,   # Required to handle the weights efficiently
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
        return content        ])
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_commands([
            "mkdir -p /root/models",
            f"curl -L -o /root/models/{filename} {download_url}"
        ])
    )

class BaseInference:
    def start_logic(self, cfg: dict):
        from llama_cpp import Llama
        model_path = f"/root/models/{cfg['filename']}"
        
        # Diagnostic check
        if not os.path.exists(model_path):
            raise RuntimeError(f"CRITICAL: File NOT found at {model_path}.")
            
        print(f"Loading {cfg.get('repo_id')} from local SSD...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=cfg["n_ctx"],
            verbose=True,    # Leave this ON to confirm 'qwen35' architecture loads
            use_mmap=True,   # Essential for larger Qwen 3.5 models
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
