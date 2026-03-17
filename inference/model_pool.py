import modal
from fleet_app import app
from inference.vllm_utils import get_tier_config, build_llama_image

cache_vol = modal.Volume.from_name("vllm-cache-vol", create_if_missing=True)

def _create_llama_cls(tier_name: str, class_name: str):
    _cfg = get_tier_config(tier_name)
    _img = build_llama_image(_cfg["model"], _cfg["filename"])
    
    @app.cls(
        image=_img, gpu=_cfg.get("gpu", "T4"), scaledown_window=2, timeout=600,
        volumes={"/root/.cache/huggingface": cache_vol}, enable_memory_snapshot=True,
    )
    class InferenceNode:
        @modal.enter(snap=True)
        def start(self):
            from llama_cpp import Llama
            self.llm = Llama.from_pretrained(repo_id=_cfg["model"], filename=_cfg["filename"], n_gpu_layers=-1, n_ctx=_cfg["n_ctx"], verbose=False)

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
            
    InferenceNode.__name__ = class_name
    return InferenceNode

InferenceSmall = _create_llama_cls("trivial", "InferenceSmall")
InferenceMedium = _create_llama_cls("simple", "InferenceMedium")
InferenceLarge = _create_llama_cls("expert", "InferenceLarge")
