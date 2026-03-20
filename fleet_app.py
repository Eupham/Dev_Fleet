import modal

app = modal.App("dev_fleet")

# Persistent volume for GGUF model files — avoids re-downloading on every cold start.
models_volume = modal.Volume.from_name("dev-fleet-models", create_if_missing=True)
