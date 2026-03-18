import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image, BaseInference

_cfg = get_tier_config("moderate")

@app.cls(
    # Add the local source directly to the image
    image=build_llama_image(**_cfg).add_local_python_source("orchestrator"),
    gpu=_cfg.get("gpu", "L40S"),
    scaledown_window=_cfg.get("scaledown_window", 2),
    timeout=_cfg.get("timeout", 1800),
)
class Inference(BaseInference):
    @modal.enter()
    def start(self):
        self.start_logic(_cfg)

    @modal.method()
    def generate(self, **kwargs):
        return self.generate_logic(**kwargs)
