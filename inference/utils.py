
"""inference/utils.py — BaseInference with tool-calling support.
FIXES APPLIED (v2):
1. generate_logic() now accepts and forwards `tools` and `tool_choice`
   to llama-server's OpenAI-compatible API.
2. When tools are present, returns the FULL ChatCompletion response object
   (not just content string) so the caller can inspect tool_calls.
3. Preserves existing schema-mode JSON parsing as fallback.
"""
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
            "ldconfig"  # Refreshes the system library cache
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
        import time
        self.start_time = time.time()
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
            "--log-disable"
        ])
        self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-run")
        timeout_secs = cfg.get("timeout", 600)
        for _ in range(timeout_secs):
            if self.server_process.poll() is not None:
                raise RuntimeError("CRITICAL: llama-server process crashed during startup.")
            try:
                resp = requests.get("http://127.0.0.1:8080/health", timeout=1)
                if resp.status_code == 200:
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"CRITICAL: llama-server failed to start within {timeout_secs} seconds.")
        print("Server is online and ready!")
    def generate_logic(self, messages, temperature=0.3, max_tokens=4096, schema=None,
                       tools=None, tool_choice=None):
        """Generate a completion, optionally with tool-calling support.
        When `tools` is provided:
        - Forwards tools/tool_choice to llama-server's OpenAI-compatible API
        - Returns the FULL ChatCompletion response object (not just content)
          so the caller can inspect .choices[0].message.tool_calls
        When `schema` is provided:
        - Uses JSON schema mode for structured output (existing behavior)
        Otherwise:
        - Returns the content string (existing behavior)
        """
        import logging
        import json
        logger = logging.getLogger("dev_fleet.inference")
        kwargs = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        # --- Tool-calling mode (NEW) ---
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            resp = self.client.chat.completions.create(**kwargs)
            # Return the full response object so caller can inspect tool_calls
            return resp
        # --- Schema mode (existing) ---
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": schema.model_json_schema()}
            }
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if schema:
            if not content or content.strip() == "":
                logger.warning(f"LLM returned empty content for schema {schema.__name__}")
                return None
            logger.info(f"Schema response (first 200 chars): {content[:200]}")
            try:
                parsed = schema.model_validate_json(content)
                logger.info(f"Successfully parsed {schema.__name__}")
                return parsed
            except Exception as e:
                logger.warning(f"Schema validation failed for {schema.__name__}: {e}")
                logger.warning(f"Raw content: {content[:1000]}")
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = schema.model_validate_json(json_match.group(1))
                        logger.info(f"Successfully parsed {schema.__name__} from code block")
                        return parsed
                    except Exception as e2:
                        logger.warning(f"Code block parsing also failed: {e2}")
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = schema.model_validate_json(json_match.group(0))
                        logger.info(f"Successfully parsed {schema.__name__} from extracted JSON")
                        return parsed
                    except Exception as e3:
                        logger.warning(f"JSON extraction also failed: {e3}")
                return None
        return content
    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
