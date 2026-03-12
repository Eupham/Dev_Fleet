# Dev Fleet — Change Log

> Running log of changes attempted during implementation.
> Reference: [Ground Truth](./ground_truth.md)

## 2026-03-12 — Unified App & Qwen3-Reranker

### Consolidate two Modal apps into one

**Goal:** Eliminate HTTP-based inter-service calls, idle-timeout waste,
and dual-credential complexity by merging `devfleet-inference` and
`devfleet-orchestrator` into a single `devfleet` app.

**Changes:**
- Created `app.py` as the single deployment entry point.
- `inference/server.py` now imports the shared `app` and adds a
  `@modal.method()` (`generate`) for Modal-native RPC.
- `orchestrator/core_app.py` imports the shared `app`; `run_agent` no
  longer requires `inference_url`.
- `orchestrator/llm_client.py`, `frege_parser.py` call
  `Inference().generate.remote()` — no httpx, no timeouts.
- Simplified `deploy.sh` to a single `modal deploy app.py` command.
- Simplified GitHub Actions workflow to a single deploy job with one
  credential pair (`DF_MODAL_TOKEN_*`).

### Replace LLM-prompting reranker with Qwen3-Reranker-0.6B

**Goal:** Use a dedicated cross-encoder model for graph-edge scoring
instead of prompting the 32B model.

**Changes:**
- Created `inference/reranker.py` — `Reranker` class loading
  `Qwen/Qwen3-Reranker-0.6B` with the official yes/no binary scoring
  pattern from the Qwen3-Reranker model card.
- Rewrote `orchestrator/rerank_engine.py` to call
  `Reranker().score_pairs.remote()` — no more httpx or LLM prompting.
- Removed `httpx` from requirements.txt, orchestrator image, and test image.

**Result:** Pending validation on Modal.

**Ground Truth Reference:** `docs/ground_truth.md`

## 2026-03-12 — Initial Implementation (Clean Start)

### Attempt 1: Full architecture scaffold

**Goal:** Build both microservices from scratch following the Modal
reference implementations and ground truth patterns.

**Changes:**
- Created `inference/server.py` — vLLM inference engine on A100-80 GB
  hosting `Qwen/Qwen2.5-Coder-32B-Instruct` with volume-cached weights.
- Created `orchestrator/core_app.py` — Modal app definition with secrets
  and deployment entrypoint.
- Created `orchestrator/graph_memory.py` — NetworkX Semantic, Procedural,
  and Episodic graph management with serialization.
- Created `orchestrator/frege_parser.py` — Pydantic schemas for atomic task
  DAG decomposition via 7B model.
- Created `orchestrator/rerank_engine.py` — Edge relevance scoring using
  cross-encoding prompts to 7B model.
- Created `orchestrator/tool_sandbox.py` — Modal Sandbox code execution
  with stdout/stderr/exit-code capture.
- Created `orchestrator/agent_loop.py` — Cyclic state machine connecting
  graph memory, parser, reranker, sandbox, and inference.
- Created `deploy.sh` — Detached deployment script.
- Created unit tests under `tests/`.

**Result:** Pending validation.

**Ground Truth Reference:** `docs/ground_truth.md`
