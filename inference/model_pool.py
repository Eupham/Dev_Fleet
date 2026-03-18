import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image, BaseInference

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

    @modal.method()
    def get_hardware_stats(self) -> dict:
        import subprocess, time
        stats = {
            "model": _cfg_small.get("repo_id", "Unknown Model"),
            "gpu": "T4",
            "uptime_sec": int(time.time() - getattr(self, "start_time", time.time())),
            "gpu_utilization": "0%", "vram_used": "0MB"
        }
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().split(',')
            if len(out) == 2:
                stats["gpu_utilization"] = f"{out[0].strip()}%"
                stats["vram_used"] = f"{out[1].strip()} MB"
        except Exception: pass
        return stats

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

    @modal.method()
    def get_hardware_stats(self) -> dict:
        import subprocess, time
        stats = {
            "model": _cfg_medium.get("repo_id", "Unknown Model"),
            "gpu": "L4",
            "uptime_sec": int(time.time() - getattr(self, "start_time", time.time())),
            "gpu_utilization": "0%", "vram_used": "0MB"
        }
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().split(',')
            if len(out) == 2:
                stats["gpu_utilization"] = f"{out[0].strip()}%"
                stats["vram_used"] = f"{out[1].strip()} MB"
        except Exception: pass
        return stats

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

    @modal.method()
    def get_hardware_stats(self) -> dict:
        import subprocess, time
        stats = {
            "model": _cfg_large.get("repo_id", "Unknown Model"),
            "gpu": "L40S",
            "uptime_sec": int(time.time() - getattr(self, "start_time", time.time())),
            "gpu_utilization": "0%", "vram_used": "0MB"
        }
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().split(',')
            if len(out) == 2:
                stats["gpu_utilization"] = f"{out[0].strip()}%"
                stats["vram_used"] = f"{out[1].strip()} MB"
        except Exception: pass
        return stats
