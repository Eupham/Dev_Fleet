# Dev Fleet — Ground Truth

> Canonical reference for architecture decisions, tested fixes, and deployment
> patterns. Every entry links to the relevant official documentation.

## Architecture Overview

Two decoupled Modal microservices communicating over HTTP:

| Service | Role | Model | GPU | Modal App Name |
|---------|------|-------|-----|----------------|
| **Inference Engine** (`inference/server.py`) | OpenAI-compatible vLLM server | `Qwen/Qwen2.5-Coder-32B-Instruct` | 1× A100-80 GB | `devfleet-inference` |
| **Orchestrator** (`orchestrator/core_app.py`) | Tri-Graph agent + sandboxed execution | `Qwen/Qwen2.5-Coder-7B-Instruct` | 1× L40S | `devfleet-orchestrator` |

## Cold-Start Mitigation (Tested)

- **Weights cached in `modal.Volume`** — never downloaded at container runtime.
  - `hf-cache-vol` — HuggingFace hub cache (`/root/.cache/huggingface`)
  - `vllm-cache-vol` — vLLM JIT compilation artifacts (`/root/.cache/vllm`)
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
- `VLLM_SERVER_DEV_MODE=1` enables sleep/wake_up endpoints.
- `TORCHINDUCTOR_COMPILE_THREADS=1` for snapshot compatibility.
- Served model alias `"llm"` so clients use a short name.

## Sandbox Execution Pattern (Tested)

- Ephemeral `modal.Sandbox.create()` per code execution.
- Captures stdout, stderr, exit code.
- Non-zero exit → new "FailedExecution" node in Episodic Graph.
- Source: <https://modal.com/docs/examples/safe_code_execution>

## Secrets

Four secrets stored in GitHub environment **"modal secrets"**:

| Secret | Purpose |
|--------|---------|
| `DF_MODAL_TOKEN_ID` | Inference service token ID |
| `DF_MODAL_TOKEN_SECRET` | Inference service token secret |
| `AG_MODAL_TOKEN_ID` | Orchestrator service token ID |
| `AG_MODAL_TOKEN_SECRET` | Orchestrator service token secret |

These are injected via `modal.Secret.from_name()` within each app definition.

## Deployment Commands

```bash
# Deploy inference engine (detached — does not block terminal)
modal deploy inference/server.py --detach

# Deploy orchestrator (detached)
modal deploy orchestrator/core_app.py --detach

# Retrieve logs
modal app logs devfleet-inference
modal app logs devfleet-orchestrator
```

## Key Dependencies

| Package | Purpose | Pinned Version |
|---------|---------|----------------|
| `vllm` | LLM inference engine | `0.13.0` |
| `huggingface_hub` | Model download | `0.36.0` |
| `networkx` | Graph memory | `>=3.2` |
| `pydantic` | Data validation / schemas | `>=2.5` |
| `httpx` | HTTP client (service-to-service) | `>=0.27` |
