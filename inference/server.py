import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image, BaseInference

_cfg = get_tier_config("moderate")

@app.cls(
    image=build_llama_image(**_cfg),
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
