import modal
import tomllib
import os
import subprocess
import time
import requests
import json
from openai import OpenAI

def clean_json_response(content: str) -> str:
    """Clean markdown formatting and trailing commentary from LLM JSON responses."""
    content = content or "{}"
    content = content.strip()

    # Try to extract content inside a markdown code block.
    # We use a non-greedy match. If there are nested backticks (e.g. valid markdown in JSON),
    # finding the outermost {} is generally much safer than relying on markdown blocks.
    # So we prefer bounding by the first '{' and the last '}' across the entire response.
    start_idx = content.find("{")
    end_idx = content.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        content = content[start_idx:end_idx+1]

    return content

def get_tier_config(tier: str) -> dict:
    """Loads the specific model configuration from config.toml."""
    with open("inference/config.toml", "rb") as f:
        config = tomllib.load(f)
    tier_cfg = config[tier]
    tier_cfg["repo_id"] = config["models"][tier]
    return tier_cfg

def build_llama_image(repo_id: str, filename: str, **kwargs) -> modal.Image:
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git", "curl", "ninja-build")
        .run_commands("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
        .env({"HF_HOME": "/vol/cache"}) 
        .pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0", "requests", "openai", "networkx>=3.2", "pydantic>=2.5")
        .run_commands([
            "git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp",
            "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs && "
            "cd /tmp/llama.cpp && "
            "cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -G Ninja && "
            "cmake --build build --config Release && "
            
            # FIX: Use cmake --install to properly place the binary AND the shared libraries
            "cmake --install build && "
            "ldconfig" # Refreshes the system library cache
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
            
            self.server_process = subprocess.Popen([
                "llama-server",
                "-m", model_path,
                "-c", str(cfg["n_ctx"]),
                "-ngl", "99",          
                "--host", "127.0.0.1",
                "--port", "8080",
                "--log-disable"  # Add this flag to suppress verbose startup logs
            ])
            
            self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-run")
            
            timeout_secs = cfg.get("timeout", 600)
            
            for _ in range(timeout_secs): 
                # 1. Fail fast if the underlying process crashed (e.g., Out of Memory)
                if self.server_process.poll() is not None:
                    raise RuntimeError("CRITICAL: llama-server process crashed during startup.")
                    
                try:
                    # Poll the health endpoint with a short timeout
                    resp = requests.get("http://127.0.0.1:8080/health", timeout=1)
                    if resp.status_code == 200:
                        break
                except (requests.ConnectionError, requests.Timeout):
                    pass
                    
                # 2. UNCONDITIONALLY sleep for 1 second if we didn't break out
                time.sleep(1)
            else:
                raise RuntimeError(f"CRITICAL: llama-server failed to start within {timeout_secs} seconds.")
                
            print("Server is online and ready!")

    def generate_logic(self, messages, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {
            "model": "local-model", 
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
            cleaned = clean_json_response(content)
            try:
                return schema.model_validate_json(cleaned)
            except Exception:
                try:
                    data = json.loads(cleaned)
                    return schema.model_validate(data)
                except json.JSONDecodeError as decode_err:
                    try:
                        data = json.loads(cleaned[:decode_err.pos])
                        return schema.model_validate(data)
                    except Exception:
                        pass

                fallback_data = {}
                for field_name, field_info in schema.model_fields.items():
                    if field_info.annotation == list or getattr(field_info.annotation, '__origin__', None) == list:
                        fallback_data[field_name] = []
                    elif field_info.annotation == dict or getattr(field_info.annotation, '__origin__', None) == dict:
                        fallback_data[field_name] = {}
                return schema.model_construct(**fallback_data)
            
        return content

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
