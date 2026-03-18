import modal
from fleet_app import app
from inference.utils import get_tier_config, build_llama_image, BaseInference

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
    def get_hardware_stats(self) -> dict:
        """
        Fetches real-time GPU hardware stats and uptime.
        """
        import subprocess
        import time
        stats = {
            "model": _cfg.get("repo_id", "Unknown Model"),
            "gpu": _cfg.get("gpu", "Unknown GPU"),
            "uptime_sec": 0,
            "gpu_utilization": "0%",
            "vram_used": "0MB"
        }
        
        # Calculate process uptime if it exists
        if hasattr(self, "start_time"):
            stats["uptime_sec"] = int(time.time() - self.start_time)

        try:
            # Query nvidia-smi for utilization and VRAM
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            out = res.stdout.strip().split(',')
            if len(out) == 2:
                stats["gpu_utilization"] = f"{out[0].strip()}%"
                stats["vram_used"] = f"{out[1].strip()} MB"
        except Exception:
            pass

        return stats

    @modal.method()
    def ping(self) -> str:
        """
        INFRASTRUCTURE FIX: A lightweight endpoint to keep the container warm 
        during the orchestrator's DRT and Kolmogorov assessment phases.
        """
        return "pong"
