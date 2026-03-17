import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image, BaseInference

# --- Trivial Tier (T4) ---
_cfg_small = get_tier_config("trivial")
@app.cls(image=build_llama_image(_cfg_small["model"], _cfg_small["filename"]), 
         gpu="T4", scaledown_window=2, timeout=600)
class InferenceSmall(BaseInference):
    def __init__(self): super().__init__(_cfg_small)
    @modal.enter()
    def start(self): self.start_logic()
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)

# --- Simple Tier (L4) ---
_cfg_medium = get_tier_config("simple")
@app.cls(image=build_llama_image(_cfg_medium["model"], _cfg_medium["filename"]), 
         gpu="L4", scaledown_window=2, timeout=600)
class InferenceMedium(BaseInference):
    def __init__(self): super().__init__(_cfg_medium)
    @modal.enter()
    def start(self): self.start_logic()
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)

# --- Expert Tier (L40S) ---
_cfg_large = get_tier_config("expert")
@app.cls(image=build_llama_image(_cfg_large["model"], _cfg_large["filename"]), 
         gpu="L40S", scaledown_window=2, timeout=1800)
class InferenceLarge(BaseInference):
    def __init__(self): super().__init__(_cfg_large)
    @modal.enter()
    def start(self): self.start_logic()
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)
