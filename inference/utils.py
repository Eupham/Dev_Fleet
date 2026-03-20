"""inference/utils.py — BaseInference with robust tool-calling support.

FIXES APPLIED (v3):
1. generate_logic() accepts and forwards `tools` and `tool_choice`.
2. When tools cause a 500 (malformed JSON from model), RETRIES WITHOUT
   tools using a structured prompt that asks for JSON action blocks instead.
   This is the critical fallback for models that struggle with native
   tool-calling format.
3. Returns full ChatCompletion when tools are used, plain content otherwise.
4. Wraps all exceptions in serializable RuntimeError so Modal RPC doesn't
   choke on openai.APIStatusError deserialization across boundaries.
5. Adds --log-disable to suppress llama-server startup noise.
"""
import modal
import tomllib
import os
import subprocess
import time
import requests
import json
import logging
import re
from openai import OpenAI

logger = logging.getLogger("dev_fleet.inference")

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
        .uv_pip_install("huggingface_hub", "langgraph>=1.1.2", "mcp>=1.26.0", "requests", "openai", "networkx>=3.2", "pydantic>=2.5")
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

# ---------------------------------------------------------------------------
# Tool definitions for the structured-fallback prompt
# ---------------------------------------------------------------------------
FALLBACK_TOOL_PROMPT = """You have these tools available. To use one, respond with EXACTLY one JSON block:
```json
{"tool": "<tool_name>", "arguments": {<args>}}
```

Available tools:
- web_search: {"query": "search terms"} — Search the web
- run_code: {"code": "python code here", "language": "python"} — Execute code
- write_file: {"path": "/workspace/file.py", "content": "file content"} — Write a file
- read_file: {"path": "/workspace/file.py"} — Read a file
- task_complete: {"summary": "what was accomplished"} — Signal task completion

IMPORTANT: Respond with ONE tool call at a time. Put code in the "code" or "content" field, never raw."""


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
        """Generate a completion with robust tool-calling support.

        Strategy:
        1. If `tools` provided, try native OpenAI tool-calling format first
        2. If llama-server returns 500 (model generated malformed tool JSON),
           fall back to structured-prompt approach (ask model to emit JSON blocks)
        3. Wrap ALL exceptions in RuntimeError for Modal RPC serialization safety
        """
        try:
            return self._generate_inner(messages, temperature, max_tokens, schema, tools, tool_choice)
        except Exception as e:
            # CRITICAL FIX: Wrap in RuntimeError so Modal can serialize it
            # across the RPC boundary. openai.APIStatusError has non-picklable
            # fields (response, body) that cause deserialization to fail.
            raise RuntimeError(f"[INFERENCE] {type(e).__name__}: {str(e)[:500]}")

    def _generate_inner(self, messages, temperature, max_tokens, schema, tools, tool_choice):
        kwargs = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # --- Tool-calling mode ---
        if tools:
            # Attempt 1: Native tool calling
            try:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
                resp = self.client.chat.completions.create(**kwargs)
                return self._serialize_response(resp)
            except Exception as e:
                error_str = str(e)
                if "500" in error_str or "parse" in error_str.lower():
                    logger.warning(f"Native tool-calling failed (model likely generated malformed JSON), "
                                   f"falling back to structured prompt. Error: {error_str[:200]}")
                    # Attempt 2: Structured prompt fallback
                    return self._fallback_tool_generation(messages, temperature, max_tokens)
                raise

        # --- Schema mode (existing) ---
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": schema.model_json_schema()}
            }

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        if schema:
            return self._parse_schema_response(content, schema)

        return content

    def _fallback_tool_generation(self, messages, temperature, max_tokens):
        """When native tool-calling fails, ask the model to emit JSON action blocks."""
        # Inject tool instructions into the system message
        fallback_messages = list(messages)

        # Find or create system message
        has_system = any(m.get("role") == "system" for m in fallback_messages)
        if has_system:
            for m in fallback_messages:
                if m["role"] == "system":
                    m["content"] = m["content"] + "\n\n" + FALLBACK_TOOL_PROMPT
                    break
        else:
            fallback_messages.insert(0, {"role": "system", "content": FALLBACK_TOOL_PROMPT})

        resp = self.client.chat.completions.create(
            model="local-model",
            messages=fallback_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""

        # Parse the JSON block from the response
        tool_call = self._extract_json_tool_call(content)

        if tool_call:
            # Return in OpenAI-compatible format as a serializable dict
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [{
                            "id": "fallback_0",
                            "type": "function",
                            "function": {
                                "name": tool_call["tool"],
                                "arguments": json.dumps(tool_call.get("arguments", {}))
                            }
                        }]
                    }
                }],
                "_fallback": True
            }

        # No tool call found, return plain content
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None
                }
            }],
            "_fallback": True
        }

    def _extract_json_tool_call(self, content: str) -> dict | None:
        """Extract a JSON tool call from model output."""
        # Try ```json blocks first
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try bare JSON objects
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content):
            try:
                parsed = json.loads(match.group(0))
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    def _serialize_response(self, resp) -> dict:
        """Convert OpenAI ChatCompletion to a plain dict for Modal RPC serialization."""
        msg = resp.choices[0].message
        result = {
            "choices": [{
                "message": {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": None
                }
            }]
        }

        if msg.tool_calls:
            result["choices"][0]["message"]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]

        return result

    def _parse_schema_response(self, content, schema):
        """Parse structured schema response with multiple fallback strategies."""
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
            logger.warning(f"Raw content: {content[:500]}")

            # Fallback: try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    parsed = schema.model_validate_json(json_match.group(1))
                    logger.info(f"Successfully parsed {schema.__name__} from code block")
                    return parsed
                except Exception as e2:
                    logger.warning(f"Code block parsing also failed: {e2}")

            # Final fallback: try to find any JSON object in the content
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = schema.model_validate_json(json_match.group(0))
                    logger.info(f"Successfully parsed {schema.__name__} from extracted JSON")
                    return parsed
                except Exception as e3:
                    logger.warning(f"JSON extraction also failed: {e3}")

            return None

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
