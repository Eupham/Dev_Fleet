"""images.py — Centralized Modal image definitions for Dev Fleet.

All container images and their package dependencies are defined here.
Consumer modules import the image they need rather than defining it inline.
"""

import modal

# ---------------------------------------------------------------------------
# Web / UI image — Chainlit frontend + orchestration helpers
# ---------------------------------------------------------------------------

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi>=0.135.1",
        "uvicorn>=0.41.0",
        "jinja2>=3.1.6",
        "python-multipart>=0.0.22",
        "pydantic>=2.12.5",
        "networkx>=3.6.1",
        "mcp>=1.26.0",
        "langgraph>=1.1.2",
        "chainlit==2.10.0",  # pinned to match the monkey-patch target
        "llama-index-core>=0.14.17",
        "llama-index-embeddings-huggingface>=0.7.0",
        "orjson>=3.11.7",
        "pathspec>=1.0.4",
        "SQLAlchemy>=2.0.0",
        "aiosqlite>=0.20.0",
    )
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .add_local_python_source("inference", copy=True)
    .add_local_python_source("ui", copy=True)
    .env({"CHAINLIT_USER_ENV": "DUMMY_ENV_TO_PREVENT_NULL_CRASH"})
    .add_local_dir(".chainlit", remote_path="/root/.chainlit", copy=True)
    .add_local_dir("public", remote_path="/root/public", copy=True)
)

# ---------------------------------------------------------------------------
# Embedder image — SentenceTransformers CPU container (Qwen3-Embedding-0.6B)
# ---------------------------------------------------------------------------

embedder_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libblas-dev", "liblapack-dev")
    .uv_pip_install(
        "scipy<1.16.0",
        "sentence-transformers>=2.0.0",
        "huggingface-hub",
        "hf_transfer",
    )
    .env({"HF_HOME": "/vol/cache"})
    .add_local_python_source("fleet_app", copy=True)
)

# ---------------------------------------------------------------------------
# Reranker image — PyTorch + Transformers CPU container (Qwen3-Reranker-0.6B)
# ---------------------------------------------------------------------------

reranker_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libblas-dev", "liblapack-dev")
    .uv_pip_install(
        "torch>=2.0",
        "transformers>=4.51.0",
        "huggingface-hub>=0.20",
        "hf_transfer",
    )
    .env({"HF_HOME": "/vol/cache"})
    .add_local_python_source("fleet_app", copy=True)
)

# ---------------------------------------------------------------------------
# Orchestrator image — CPU container for the tri-graph agent loop
# ---------------------------------------------------------------------------

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("inference", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .apt_install("libblas-dev", "liblapack-dev")
    .uv_pip_install(
        "scipy<1.16.0",
        "networkx>=3.2",
        "pydantic>=2.5",
        "pathspec>=0.12.1",
        "orjson>=3.11.7",
        "llama-index-core>=0.10.0",
        "llama-index>=0.10.0",
        "llama-index-embeddings-huggingface>=0.1.0",
        "langgraph>=1.1.2",
        "mcp>=1.26.0",
        "radon>=6.0",
        "tree-sitter>=0.25.2",
        "tree-sitter-javascript>=0.23",
        "markdown-it-py>=3.0",
        "beautifulsoup4>=4.12",
        "trafilatura>=2.0.0",
        "pymupdf>=1.27.2",
        "ddgs>=6.3.7",           # web_search tool — baked in so no runtime install
    )
)

# ---------------------------------------------------------------------------
# Llama image builder — GPU container with llama.cpp + CUDA (parameterised)
# ---------------------------------------------------------------------------

def build_llama_image(repo_id: str, filename: str, **kwargs) -> modal.Image:
    """Build a GPU inference image for a specific GGUF model.

    The model file is NOT baked into the image — it is downloaded on first cold start
    via huggingface_hub + hf_transfer and persisted to the ``dev-fleet-models`` Volume.
    """
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "clang", "cmake", "git", "curl", "ninja-build")
        .run_commands("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1")
        .env({"HF_HOME": "/vol/cache"})
        .uv_pip_install(
            "huggingface_hub",
            "hf_transfer",
            "langgraph>=1.1.2",
            "mcp>=1.26.0",
            "requests",
            "openai",
            "networkx>=3.2",
            "pydantic>=2.5",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .run_commands([
            "git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp",
            "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs && "
            "cd /tmp/llama.cpp && "
            "cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -G Ninja && "
            "cmake --build build --config Release && "
            "cmake --install build && "
            "ldconfig",
        ])
        .add_local_python_source("fleet_app", copy=True)
        .add_local_file("inference/config.toml", remote_path="/root/inference/config.toml", copy=True)
        .run_commands(["mkdir -p /root/models"])
    )
