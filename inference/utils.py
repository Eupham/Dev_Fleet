import modal
import tomllib  # Use the built-in Python 3.11+ library
import os

def get_tier_config(tier: str) -> dict:
    """Loads the specific model configuration from config.toml."""
    # tomllib requires the file to be opened in binary mode ("rb")
    with open("inference/config.toml", "rb") as f:
        config = tomllib.load(f)
    tier_cfg = config[tier]
    tier_cfg["repo_id"] = config["models"][tier]
    return tier_cfg

def build_llama_image(repo_id: str, filename: str, **kwargs) -> modal.Image:
    # ... (keep the rest of your build_llama_image function exactly as is)
