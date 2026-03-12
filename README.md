# Dev Fleet

> Modal-hosted autonomous coding agent — an open-weight alternative to
> OpenHands/Devin using Qwen models exclusively.

## Architecture

Two decoupled microservices running on [Modal](https://modal.com):

| Service | Description | Model | GPU |
|---------|-------------|-------|-----|
| **Inference Engine** | OpenAI-compatible vLLM server | `Qwen/Qwen2.5-Coder-32B-Instruct` | A100-80 GB |
| **Orchestrator** | Tri-Graph agent + sandboxed execution | `Qwen/Qwen2.5-Coder-7B-Instruct` | L40S |

```
User Prompt
    │
    ▼
┌──────────────────────────┐
│  Orchestrator (Service B)│
│  ┌─────────────────────┐ │        ┌──────────────────────┐
│  │ Frege Parser (7B)   │─┼──HTTP──▶  Inference Engine    │
│  │ Rerank Engine (7B)  │ │        │  (Service A)          │
│  │ Graph Memory (NX)   │ │        │  vLLM + 32B model     │
│  │ Agent Loop           │ │        │  /v1/chat/completions │
│  │ Tool Sandbox         │ │        └──────────────────────┘
│  └─────────────────────┘ │
└──────────────────────────┘
```

## Project Structure

```
Dev_Fleet/
├── inference/
│   └── server.py            # Microservice A — vLLM on A100
├── orchestrator/
│   ├── core_app.py          # Modal app, secrets, entrypoint
│   ├── graph_memory.py      # Semantic / Procedural / Episodic graphs (NetworkX)
│   ├── frege_parser.py      # Prompt → Task DAG decomposition (Pydantic)
│   ├── rerank_engine.py     # Cross-encode edge relevance scoring
│   ├── tool_sandbox.py      # Ephemeral Modal Sandbox execution
│   └── agent_loop.py        # Cyclic state machine
├── tests/                   # Unit tests (pytest)
├── docs/
│   ├── ground_truth.md      # Canonical architecture decisions & tested patterns
│   └── change_log.md        # Running log of changes
├── deploy.sh                # Detached deployment script
└── requirements.txt         # Local development dependencies
```

## Quick Start

### Prerequisites

- Python 3.12+
- A Modal account with secrets configured (see below)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Modal Secrets

Modal authentication is handled via GitHub environment secrets in the
`modal secrets` environment.  The workflow sets `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` automatically — no Modal-side secret group is needed.

| GitHub Secret | Purpose |
|---------------|---------|
| `DF_MODAL_TOKEN_ID` | Inference service token ID |
| `DF_MODAL_TOKEN_SECRET` | Inference service token secret |
| `AG_MODAL_TOKEN_ID` | Orchestrator service token ID |
| `AG_MODAL_TOKEN_SECRET` | Orchestrator service token secret |

### Deploy

```bash
# Deploy both services (detached — non-blocking)
./deploy.sh

# Or deploy individually
./deploy.sh inference
./deploy.sh orchestrator
```

### View Logs

```bash
modal app logs devfleet-inference
modal app logs devfleet-orchestrator
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Design Principles

- **Frege's Compositionality**: Complex prompts are decomposed into a DAG of
  atomic tasks using a small (7B) model, then executed independently.
- **Tri-Graph Memory**: Three NetworkX digraphs (Semantic, Procedural,
  Episodic) provide structured context for every LLM call.
- **Sandboxed Execution**: All code runs in ephemeral Modal Sandboxes — the
  host container is never at risk.
- **Cold-Start Mitigation**: GPU memory snapshots serialize CPU+GPU state
  after warmup so containers restore instantly — no JIT compilation needed.
  The GPU scales to zero when idle.