import modal
from fleet_app import app
from inference.vllm_utils import get_tier_config, build_llama_image

_cfg = get_tier_config("moderate")
_image = build_llama_image(_cfg["model"], _cfg["filename"])
cache_vol = modal.Volume.from_name("vllm-cache-vol", create_if_missing=True)

@app.cls(
    image=_image,
    gpu=_cfg.get("gpu", "L40S"),
    scaledown_window=2,
    timeout=1800,
    volumes={"/root/.cache/huggingface": cache_vol},
    enable_memory_snapshot=True,
)
class Inference:
    @modal.enter(snap=True)
    def start(self):
        print(f"[dev_fleet] Loading {_cfg['model']} into VRAM...")
        from llama_cpp import Llama
        self.llm = Llama.from_pretrained(
            repo_id=_cfg["model"],
            filename=_cfg["filename"],
            n_gpu_layers=-1, 
            n_ctx=_cfg["n_ctx"],
            verbose=False
        )
        print("[dev_fleet] Model loaded. Snapshot ready.")

    @modal.method()
    def generate(self, messages, model=None, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": schema.model_json_schema()}
            }
        resp = self.llm.create_chat_completion(**kwargs)
        content = resp["choices"][0]["message"]["content"]
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content
