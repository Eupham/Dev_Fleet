import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image, BaseInference

_cfg = get_tier_config("moderate")

@app.cls(
    image=build_llama_image(_cfg["model"], _cfg["filename"]),
    gpu=_cfg.get("gpu", "L40S"),
    scaledown_window=_cfg.get("scaledown_window", 2),
    timeout=_cfg.get("timeout", 1800),
)
class Inference(BaseInference):
    def __init__(self):
        super().__init__(_cfg)

    @modal.enter()
    def start(self):
        self.start_logic()

    @modal.method()
    def generate(self, **kwargs):
        return self.generate_logic(**kwargs)
