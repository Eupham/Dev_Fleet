"""Model Pool — Tier-based inference classes for difficulty routing."""
from __future__ import annotations
from typing import Any, Optional
import modal
from fleet_app import app
from inference.vllm_utils import get_tier_config, build_vllm_image, wait_for_vllm, build_serve_cmd

# ---------------------------------------------------------------------------
# Trivial tier
# ---------------------------------------------------------------------------
_cfg_trivial = get_tier_config("trivial")
_trivial_image = build_vllm_image(_cfg_trivial["model"], is_nightly=False)

with _trivial_image.imports():
    import requests as _requests_small

@app.cls(
    image=_trivial_image, gpu="T4", scaledown_window=2, timeout=10 * 60, retries=0,
    enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=50)
class InferenceSmall:
    @modal.enter(snap=True)
    def start(self):
        import subprocess, os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.proc = subprocess.Popen(build_serve_cmd(_cfg_trivial))
        wait_for_vllm(self.proc, _cfg_trivial["port"])
        _requests_small.post(
            f"http://localhost:{_cfg_trivial['port']}/v1/chat/completions",
            json={"model": _cfg_trivial['served_model_name'], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=120,
        )
        _requests_small.post(f"http://localhost:{_cfg_trivial['port']}/sleep?level=1", timeout=60).raise_for_status()

    @modal.enter(snap=False)
    def restore(self):
        _requests_small.post(f"http://localhost:{_cfg_trivial['port']}/wake_up", timeout=120).raise_for_status()

    @modal.method()
    def generate(self, messages: list[dict], model: str = None, temperature: float = 0.3, max_tokens: int = 2048, schema: Optional[Any] = None) -> Any:
        payload = {"model": model or _cfg_trivial['served_model_name'], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
            payload["tool_choice"] = "none"
        resp = _requests_small.post(f"http://localhost:{_cfg_trivial['port']}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content

    @modal.exit()
    def stop(self): self.proc.terminate()

# ---------------------------------------------------------------------------
# Simple tier
# ---------------------------------------------------------------------------
_cfg_simple = get_tier_config("simple")
_simple_image = build_vllm_image(_cfg_simple["model"], is_nightly=False)

with _simple_image.imports():
    import requests as _requests_medium

@app.cls(
    image=_simple_image, gpu="A10G", scaledown_window=2, timeout=10 * 60, retries=0,
    enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=30)
class InferenceMedium:
    @modal.enter(snap=True)
    def start(self):
        import subprocess, os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.proc = subprocess.Popen(build_serve_cmd(_cfg_simple))
        wait_for_vllm(self.proc, _cfg_simple["port"])
        _requests_medium.post(
            f"http://localhost:{_cfg_simple['port']}/v1/chat/completions",
            json={"model": _cfg_simple['served_model_name'], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=120,
        )
        _requests_medium.post(f"http://localhost:{_cfg_simple['port']}/sleep?level=1", timeout=60).raise_for_status()

    @modal.enter(snap=False)
    def restore(self):
        _requests_medium.post(f"http://localhost:{_cfg_simple['port']}/wake_up", timeout=120).raise_for_status()

    @modal.method()
    def generate(self, messages: list[dict], model: str = None, temperature: float = 0.3, max_tokens: int = 4096, schema: Optional[Any] = None) -> Any:
        payload = {"model": model or _cfg_simple['served_model_name'], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
            payload["tool_choice"] = "none"
        resp = _requests_medium.post(f"http://localhost:{_cfg_simple['port']}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content

    @modal.exit()
    def stop(self): self.proc.terminate()

# ---------------------------------------------------------------------------
# Expert tier
# ---------------------------------------------------------------------------
_cfg_expert = get_tier_config("expert")
_expert_image = build_vllm_image(_cfg_expert["model"], is_nightly=False)

with _expert_image.imports():
    import requests as _requests_large

@app.cls(
    image=_expert_image, gpu="A100-80GB", scaledown_window=2, timeout=20 * 60, retries=0,
    enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=10)
class InferenceLarge:
    @modal.enter(snap=True)
    def start(self):
        import subprocess, os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.proc = subprocess.Popen(build_serve_cmd(_cfg_expert))
        wait_for_vllm(self.proc, _cfg_expert["port"])
        _requests_large.post(
            f"http://localhost:{_cfg_expert['port']}/v1/chat/completions",
            json={"model": _cfg_expert['served_model_name'], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=300,
        )
        _requests_large.post(f"http://localhost:{_cfg_expert['port']}/sleep?level=1", timeout=120).raise_for_status()

    @modal.enter(snap=False)
    def restore(self):
        _requests_large.post(f"http://localhost:{_cfg_expert['port']}/wake_up", timeout=180).raise_for_status()

    @modal.method()
    def generate(self, messages: list[dict], model: str = None, temperature: float = 0.3, max_tokens: int = 8192, schema: Optional[Any] = None) -> Any:
        payload = {"model": model or _cfg_expert['served_model_name'], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
            payload["tool_choice"] = "none"
        resp = _requests_large.post(f"http://localhost:{_cfg_expert['port']}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content

    @modal.exit()
    def stop(self): self.proc.terminate()
