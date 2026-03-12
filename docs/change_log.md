# Dev Fleet — Change Log

> Running log of changes attempted during implementation.
> Reference: [Ground Truth](./ground_truth.md)

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
