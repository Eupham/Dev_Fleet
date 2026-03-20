import modal
from fleet_app import app
from images import build_llama_image
from inference.utils import get_tier_config, BaseInference

_cfg = get_tier_config("moderate")

@app.cls(
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
        
    @modal.method()
    def ping(self) -> str:
        """
        INFRASTRUCTURE FIX: A lightweight endpoint to keep the container warm 
        during the orchestrator's DRT and Kolmogorov assessment phases.
        """
        return "pong"
