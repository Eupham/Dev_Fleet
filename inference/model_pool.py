import modal
from fleet_app import app
from inference.utils import BaseInference, get_tier_config, build_llama_image

# Logic is defined ONCE in BaseInference (in utils.py). 
# These classes only define the "Shell" (The Hardware).

@app.cls(gpu="T4", image=build_llama_image(**get_tier_config("trivial")))
class InferenceSmall(BaseInference):
    @modal.enter()
    def start(self): self.start_logic()
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)

@app.cls(gpu="L4", image=build_llama_image(**get_tier_config("simple")))
class InferenceMedium(BaseInference):
    @modal.enter()
    def start(self): self.start_logic()
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)

@app.cls(gpu="L40S", image=build_llama_image(**get_tier_config("expert")))
class InferenceLarge(BaseInference):
    @modal.enter()
    def start(self): self.start_logic()
    @modal.method()
    def generate(self, **kwargs): return self.generate_logic(**kwargs)
