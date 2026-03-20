import modal
from fleet_app import app
from images import build_llama_image
from inference.utils import get_tier_config, BaseInference

# --- Trivial Tier (T4) ---
_cfg_small = get_tier_config("trivial")
@app.cls(
    image=build_llama_image(**_cfg_small).add_local_python_source("orchestrator"), 
    gpu="T4", 
    scaledown_window=2, 
    timeout=600
)
class InferenceSmall(BaseInference):
    @modal.enter()
    def start(self): self.start_logic(_cfg_small)
    
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)

# --- Simple Tier (L4) ---
_cfg_medium = get_tier_config("simple")
@app.cls(
    image=build_llama_image(**_cfg_medium).add_local_python_source("orchestrator"), 
    gpu="L4", 
    scaledown_window=2, 
    timeout=600
)
class InferenceMedium(BaseInference):
    @modal.enter()
    def start(self): self.start_logic(_cfg_medium)
    
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)

# --- Expert Tier (L40S) ---
_cfg_large = get_tier_config("expert")
@app.cls(
    image=build_llama_image(**_cfg_large).add_local_python_source("orchestrator"), 
    gpu="L40S", 
    scaledown_window=2, 
    timeout=1800
)
class InferenceLarge(BaseInference):
    @modal.enter()
    def start(self): self.start_logic(_cfg_large)
    
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)
