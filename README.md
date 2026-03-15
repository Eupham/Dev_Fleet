# Dev Fleet

> Modal-hosted autonomous coding agent — an open-weight alternative to
> OpenHands/Devin using Qwen models exclusively.

## Architecture

A single Modal app (`dev_fleet`) with three container types:

| Container | Description | Model | Resource |
|-----------|-------------|-------|----------|
| **Inference** | OpenAI-compatible vLLM server | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | L40S |
| **Reranker** | Cross-encoder edge scoring | `Qwen/Qwen3-Reranker-0.6B` | CPU |
| **Orchestrator** | Tri-Graph agent + sandboxed execution | — | CPU |

All inter-container calls use **Modal-native RPC** (``.remote()``)
within the same app.

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  dev_fleet (single Modal app)                         │
│                                                      │
│  ┌─────────────────────┐                             │
│  │ Orchestrator (CPU)  │                             │
│  │  Frege Parser       │──.remote()──▶ Inference     │
│  │  Graph Memory (NX)  │              (A100, vLLM)   │
│  │  Agent Loop         │                             │
│  │  Tool Sandbox       │──.remote()──▶ Reranker      │
│  └─────────────────────┘              (CPU, 0.6B)    │
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
Dev_Fleet/
├── app.py                   # Unified Modal app entry point
├── inference/
│   ├── server.py            # Inference — vLLM on A100 (+ web endpoint)
│   └── reranker.py          # Reranker — Qwen3-Reranker-0.6B
├── orchestrator/
│   ├── core_app.py          # run_agent function (CPU)
│   ├── graph_memory.py      # Semantic / Procedural / Episodic graphs (NetworkX)
│   ├── frege_parser.py      # Prompt → Task DAG decomposition (Pydantic)
│   ├── rerank_engine.py     # Cross-encoder edge scoring logic
│   ├── llm_client.py        # Thin wrapper for Modal-native inference calls
│   ├── tool_sandbox.py      # Ephemeral Modal Sandbox execution
│   └── agent_loop.py        # Cyclic state machine
├── tests/                   # Unit tests (Modal-native)
├── docs/
│   ├── ground_truth.md      # Canonical architecture decisions & tested patterns
│   └── change_log.md        # Running log of changes
├── deploy.sh                # Single deployment script
└── requirements.txt         # Local development dependencies
```

## Quick Start

### Prerequisites

- Python 3.12+
- A Modal account

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Modal Secrets

Modal authentication is handled via GitHub environment secrets in the
`modal secrets` environment.  The workflow sets `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` automatically.

| GitHub Secret | Purpose |
|---------------|---------|
| `DF_MODAL_TOKEN_ID` | Modal token ID |
| `DF_MODAL_TOKEN_SECRET` | Modal token secret |

### Deploy

```bash
# Deploy the entire app (inference + reranker + orchestrator)
./deploy.sh

# Or directly
modal deploy app.py
```

### View Logs

```bash
modal app logs dev_fleet
```

### Run Tests

```bash
modal run tests/modal_test_runner.py
```

## Design Principles

- **Frege's Compositionality**: Complex prompts are decomposed into a DAG of
  atomic tasks using the 7B model. Each task declares `inputs_needed` and
  `outputs_produced` so the DAG validator can check that the composition is
  correctly ordered. Tasks are then scored against knowledge-graph nodes using
  the Qwen3-Reranker.
- **Tri-Graph Memory**: Three NetworkX digraphs (Semantic, Procedural,
  Episodic) provide structured context for every LLM call.
- **Dedicated Reranker**: Qwen3-Reranker-0.6B cross-encoder assesses
  relevance between task DAG nodes and knowledge-graph nodes, without LLM
  prompting for scoring.
- **Modal-Native RPC**: Inter-container calls use `.remote()` within the
  same app.
- **Sandboxed Execution**: Code runs in ephemeral Modal Sandboxes, isolated
  from the orchestrator container.
- **Cold-Start Mitigation**: GPU memory snapshots are used to reduce container
  restore time. The GPU scales to zero when idle.