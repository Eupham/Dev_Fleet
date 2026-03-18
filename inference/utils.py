import modal
import tomllib
import os
import subprocess
import time
import requests
from openai import OpenAI

def get_tier_config(tier: str) -> dict:
    """Loads the specific model configuration from config.toml."""
    with open("inference/config.toml", "rb") as f:
        config = tomllib.load(f)
    tier_cfg = config[tier]
    tier_cfg["repo_id"] = config["models"][tier]
    return tier_cfg

def build_llama_image(repo_id: str, filename: str, **kwargs) -> modal.Image:
    """
    Directly compiles the official upstream llama.cpp server to ensure
    support for the newest architectures like Qwen 3.5.
    """
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git", "curl", "ninja-build")
        .run_commands("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
        .env({"HF_HOME": "/vol/cache"}) 
        # Added requests and openai for the HTTP client
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0", "requests", "openai")
        .run_commands([
            # 1. Clone the master branch of llama.cpp directly
            "git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp",
            
            # 2. Compile the raw server with CUDA support
            "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs && "
            "cd /tmp/llama.cpp && "
            "cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -G Ninja && "
            "cmake --build build --config Release && "
            
            # 3. Move the binary to our PATH
            "find build -name llama-server -exec cp {} /usr/local/bin/ \\;"
        ])
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_commands([
            "mkdir -p /root/models",
            f"curl -L -o /root/models/{filename} {download_url}"
        ])
    )

class BaseInference:
    def start_logic(self, cfg: dict):
        model_path = f"/root/models/{cfg['filename']}"
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"CRITICAL: File NOT found at {model_path}.")
            
        print(f"Booting raw llama-server for {cfg.get('repo_id')}...")
        
        # Start the native C++ server in the background
        self.server_process = subprocess.Popen([
            "llama-server",
            "-m", model_path,
            "-c", str(cfg["n_ctx"]),
            "-ngl", "99",          # Offload all layers to GPU
            "--host", "127.0.0.1",
            "--port", "8080"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Poll until the server is responsive
        self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-run")
        
        for _ in range(60): # 60-second timeout
            try:
                if requests.get("http://127.0.0.1:8080/health").status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("CRITICAL: llama-server failed to start within 60 seconds.")
            
        print("Server is online and ready!")

    def generate_logic(self, messages, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {
            "model": "local-model", # The server ignores this string, but OpenAI client requires it
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema", 
                "json_schema": {"schema": schema.model_json_schema()}
            }
            
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except: return schema.model_construct()
            
        return content

    def __del__(self):
        # Ensure the background C++ process is killed when the container winds down
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
