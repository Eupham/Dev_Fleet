import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image

cache_vol = modal.Volume.from_name("model-cache-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Trivial tier
# ---------------------------------------------------------------------------
_cfg_small = get_tier_config("trivial")
_img_small = build_llama_image(_cfg_small["model"], _cfg_small["filename"])

@app.cls(
    image=_img_small, gpu=_cfg_small.get("gpu", "T4"), scaledown_window=2, timeout=600,
    volumes={"/vol/cache": cache_vol}, enable_memory_snapshot=True,
)
class InferenceSmall:
    @modal.enter(snap=True)
        def start(self):
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            
            model_path = hf_hub_download(
                repo_id=_cfg_small["model"], 
                filename=_cfg_small["filename"]
            )
            
            self.llm = Llama(
                model_path=model_path, 
                n_gpu_layers=-1, 
                n_ctx=_cfg_small["n_ctx"], 
                verbose=False
            )
    @modal.method()
    def generate(self, messages, model=None, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema: kwargs["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema.model_json_schema()}}
        resp = self.llm.create_chat_completion(**kwargs)
        content = resp["choices"][0]["message"]["content"]
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content

# ---------------------------------------------------------------------------
# Simple tier
# ---------------------------------------------------------------------------
_cfg_medium = get_tier_config("simple")
_img_medium = build_llama_image(_cfg_medium["model"], _cfg_medium["filename"])

@app.cls(
    image=_img_medium, gpu=_cfg_medium.get("gpu", "L4"), scaledown_window=2, timeout=600,
    volumes={"/vol/cache": cache_vol}, enable_memory_snapshot=True,
)
class InferenceMedium:
    @modal.enter(snap=True)
    def start(self):
        from llama_cpp import Llama
        self.llm = Llama.from_pretrained(repo_id=_cfg_medium["model"], filename=_cfg_medium["filename"], n_gpu_layers=-1, n_ctx=_cfg_medium["n_ctx"], verbose=False)

    @modal.method()
    def generate(self, messages, model=None, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema: kwargs["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema.model_json_schema()}}
        resp = self.llm.create_chat_completion(**kwargs)
        content = resp["choices"][0]["message"]["content"]
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content

# ---------------------------------------------------------------------------
# Expert tier
# ---------------------------------------------------------------------------
_cfg_large = get_tier_config("expert")
_img_large = build_llama_image(_cfg_large["model"], _cfg_large["filename"])

@app.cls(
    image=_img_large, gpu=_cfg_large.get("gpu", "L40S"), scaledown_window=2, timeout=1800,
    volumes={"/vol/cache": cache_vol}, enable_memory_snapshot=True,
)
class InferenceLarge:
    @modal.enter(snap=True)
    def start(self):
        from llama_cpp import Llama
        self.llm = Llama.from_pretrained(repo_id=_cfg_large["model"], filename=_cfg_large["filename"], n_gpu_layers=-1, n_ctx=_cfg_large["n_ctx"], verbose=False)

    @modal.method()
    def generate(self, messages, model=None, temperature=0.3, max_tokens=4096, schema=None):
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema: kwargs["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema.model_json_schema()}}
        resp = self.llm.create_chat_completion(**kwargs)
        content = resp["choices"][0]["message"]["content"]
        if schema:
            try: return schema.model_validate_json(content or "{}")
            except Exception: return schema.model_construct()
        return content
