import modal
from fleet_app import app
from inference.utils import build_llama_image, get_tier_config

_cfg = get_tier_config("trivial")
image = build_llama_image(**_cfg).pip_install("toml").add_local_dir("inference", remote_path="/root/inference", copy=True)

@app.function(image=image, gpu="T4")
def test_load():
    import os
    import llama_cpp
    from llama_cpp import Llama

    print("llama_cpp version:", llama_cpp.__version__)

    model_path = f"/root/models/{_cfg['filename']}"
    print("Size:", os.path.getsize(model_path))
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=-1)
        print("Success loading Qwen3.5 with custom patched llama.cpp!")
    except Exception as e:
        print("Failed:", e)
