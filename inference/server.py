import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image

_cfg = get_tier_config("moderate")
_image = build_llama_image(_cfg["model"], _cfg["filename"])
cache_vol = modal.Volume.from_name("model-cache-vol", create_if_missing=True)

@app.cls(
    image=_image,
    gpu=_cfg.get("gpu", "L40S"),
    scaledown_window=2,
    timeout=1800,
    volumes={"/vol/cache": cache_vol},
    enable_memory_snapshot=True,
)
class Inference:
    @modal.enter(snap=True)
    def start(self):
        print(f"[dev_fleet] Ensuring {_cfg['model']} is present in Volume...")
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        
        # Explicitly download to get the exact path, avoiding 'from_pretrained' issues
        model_path = hf_hub_download(
            repo_id=_cfg["model"],
            filename=_cfg["filename"],
        )
        
        print(f"[dev_fleet] Loading model from {model_path} into VRAM...")
        self.llm = Llama(
            model_path=model_path,
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
