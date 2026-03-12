# Dev Fleet — Ground Truth

> Canonical reference for architecture decisions, tested fixes, and deployment
> patterns. Every entry links to the relevant official documentation.

## Architecture Overview

A single Modal app (`devfleet`) with three container types communicating
via Modal-native RPC (`.remote()`):

| Container | Role | Model | Resource | File |
|-----------|------|-------|----------|------|
| **Inference** | OpenAI-compatible vLLM server | `Qwen/Qwen2.5-Coder-32B-Instruct` | 1× A100-80 GB | `inference/server.py` |
| **Reranker** | Cross-encoder edge scoring | `Qwen/Qwen3-Reranker-0.6B` | CPU | `inference/reranker.py` |
| **Orchestrator** | Tri-Graph agent + sandboxed execution | — | CPU | `orchestrator/core_app.py` |

## Cold-Start Mitigation (Tested)

- **Weights cached in `modal.Volume`** — never downloaded at container runtime.
  - `hf-cache-vol` — HuggingFace hub cache (`/root/.cache/huggingface`)
  - `vllm-cache-vol` — vLLM JIT compilation artifacts (`/root/.cache/vllm`)
  - `reranker-cache-vol` — Qwen3-Reranker weights (`/root/.cache/huggingface`)
- **GPU Memory Snapshots** — `enable_memory_snapshot=True` and
  `experimental_options={"enable_gpu_snapshot": True}`.  The vLLM server is
  started, warmed up, then put to sleep before the snapshot.  On restore the
  server wakes instantly — no JIT compilation needed.  **No `min_containers`
  / `keep_warm` — the GPU scales to zero.**
- **`HF_HUB_ENABLE_HF_TRANSFER=1`** environment variable for fast parallel
  downloads during the *build* step only.
- Source: <https://modal.com/docs/guide/memory-snapshots>
- vLLM snapshot example: <https://modal.com/docs/examples/vllm_snapshot>

## vLLM Serving Pattern (Tested)

- Based on official Modal vLLM snapshot example: <https://modal.com/docs/examples/vllm_snapshot>
- Uses `@app.cls()` with GPU snapshot lifecycle:
  - `@modal.enter(snap=True)`: start vLLM, warm up with sample requests,
    then call `/sleep` to offload weights to CPU before snapshot.
  - `@modal.enter(snap=False)`: call `/wake_up` to reload weights on restore.
  - `@modal.exit()`: terminate the subprocess.
- `@modal.method()` (`generate`): Modal-native RPC for orchestrator calls.
- `@modal.web_server()`: external OpenAI-compatible HTTP endpoint.
- `VLLM_SERVER_DEV_MODE=1` enables sleep/wake_up endpoints.
- `TORCHINDUCTOR_COMPILE_THREADS=1` for snapshot compatibility.
- Served model alias `"llm"` so clients use a short name.

## Reranker Pattern

- Uses `Qwen/Qwen3-Reranker-0.6B` cross-encoder model.
- Binary yes/no judgment converted to [0, 1] relevance score via
  log-softmax over token logits.
- Assesses relevance between task DAG nodes and knowledge-graph nodes,
  following Frege's compositionality principle to derive task complexity.
- Runs on CPU (0.6B model is lightweight).
- Source: <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>

## Sandbox Execution Pattern (Tested)

- Ephemeral `modal.Sandbox.create()` per code execution.
- Captures stdout, stderr, exit code.
- Non-zero exit → new "FailedExecution" node in Episodic Graph.
- Source: <https://modal.com/docs/examples/safe_code_execution>

## Secrets

GitHub environment **"modal secrets"** — only one token pair needed
for the unified app:

| Secret | Purpose |
|--------|---------|
| `DF_MODAL_TOKEN_ID` | Modal token ID |
| `DF_MODAL_TOKEN_SECRET` | Modal token secret |

These are injected as `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment
variables by the GitHub Actions workflow.

## Deployment Commands

```bash
# Deploy the entire app
modal deploy app.py

# Retrieve logs
modal app logs devfleet

# Quick test
modal run app.py --prompt "Write a hello world function"
```

## Key Dependencies

| Package | Purpose | Pinned Version |
|---------|---------|----------------|
| `vllm` | LLM inference engine | `0.13.0` |
| `huggingface_hub` | Model download | `0.36.0` |
| `transformers` | Qwen3-Reranker model loading | `>=4.51.0` |
| `torch` | Reranker inference | `>=2.0` |
| `networkx` | Graph memory | `>=3.2` |
| `pydantic` | Data validation / schemas | `>=2.5` |
