# Project Initiative: `dev_fleet` & `Alt_git` Foundation



## 🎯 Overarching Objective
Bootstrap a fully isolated, headless application and version control ecosystem on Modal from scratch.

This project bypasses standard GitHub infrastructure entirely.
* **Workspace 1 (`Alt_git`):** A custom, lightweight version control backend API.
* **Workspace 2 (`dev_fleet`):** Hosts the main application and a dedicated AI maintenance toolset (`fleet_fix`).
* **Workspace 3 (`Jules_Test_Lab`):** An isolated testing sandbox.

The ultimate goal is a self-healing codebase managed entirely by a local fleet of Qwen models.

## 🧭 The Core Philosophy: "No Blind Coding"
We are building from zero. Every piece of code must be grounded in verified facts.
1. **Research First:** Read offline documentation from the Modal Volume before coding.
2. **Establish Ground Truth:** Synthesize findings into `docs/issues/{issue_id}_ground_truth.md` inside `dev_fleet`.
3. **Maintain the Diary:** Log the attempt, reasoning, and outcome in `docs/issues/{issue_id}_change_log_diary.md` inside `dev_fleet`.

## ⚖️ Architectural Guidelines: Anti-Monolith & Frameworks
To avoid monolithic spaghetti code and redundant boilerplate, strict Separation of Concerns (SoC) is enforced:
1. **Strict Boundary Enforcement:** `dev_fleet` must *never* directly manipulate files in `Alt_git`. All state changes, file reads, and commits must happen strictly through HTTP requests to the `Alt_git` API endpoints.
2. **Framework over Boilerplate:** Do not write custom HTTP server loops or manual payload parsers. Use **FastAPI** to define endpoints and **Pydantic** to validate incoming JSON payloads automatically.
3. **Isolate Compute Profiles:** Pure execution routing runs on Modal CPU instances. Qwen generation must be separated into GPU-specific Modal functions (`@app.function(gpu="A100")`) using **vLLM**.

## 📚 Open Source Libraries & Frameworks
Use the following tools to construct the system.

**Core Infrastructure (`Alt_git` / `dev_fleet`):**
* **Modal Docs:** https://modal.com/docs/guide
* **FastAPI:** https://fastapi.tiangolo.com/
* **Pydantic:** https://docs.pydantic.dev/
* **DeepDiff:** https://zepworks.com/deepdiff/current/

**AI Orchestration & Parsing (`dev_fleet` / `fleet_fix`):**
* **vLLM:** https://docs.vllm.ai/en/latest/
* **LiteLLM:** https://docs.litellm.ai/docs/
* **DSPy:** https://dspy-docs.vercel.app/docs/intro
* **grep-ast:** https://github.com/paul-gauthier/grep-ast
* **LibCST:** https://libcst.readthedocs.io/
* **Qdrant:** https://qdrant.tech/documentation/
* **NetworkX:** https://networkx.org/documentation/stable/

**AI-Native Ingestion (Phase 3 Targets):**
* **Qwen3 Embedding:** https://huggingface.co/collections/Qwen/qwen3-embedding
* **Qwen3 Reranker:** https://huggingface.co/collections/Qwen/qwen3-reranker

## 🗂️ Proposed File Structures

### Workspace 1: `Alt_git`
```text
alt_git/
├── api/
│   ├── routes.py          # FastAPI endpoints (/telemetry/crash, /commit, /repo/current)
│   └── dependencies.py    # Auth validation using AG_MODAL_TOKEN secrets
├── core/
│   ├── versioning.py      # hashlib, difflib, and DeepDiff history tracking
│   └── storage.py         # Modal Volume interactions for file persistence
├── schemas/
│   └── payloads.py        # Pydantic models (FileChange, CommitPayload, RepoStateResponse)
├── modal_app.py           # Modal @app.function definitions and ASGI mounts
└── requirements.txt       # fastapi, pydantic, deepdiff
```

### Workspace 2: `dev_fleet`
```text
dev_fleet/
├── app/                   # Headless core application
│   ├── main.py            # Trigger endpoints
│   └── worker.py          # Execution logic wrapped in global try/except
├── fleet_fix/             # AI maintenance toolset
│   ├── orchestrator.py    # Catches tracebacks, triggers DSPy pipelines
│   ├── alt_git_client.py  # HTTP REST wrapper for Alt_git API calls
│   ├── state/             # Handles Ground Truth and Diary read/writes
│   ├── parsers/           # grep-ast and libcst integrations
│   └── llm_router.py      # LiteLLM routing (Jules API -> local vLLM Qwen)
├── schemas/
│   └── states.py          # Pydantic (CrashReport, AgentStateLog)
├── doc_ingestion.py       # Knowledge base builder
├── modal_app.py           # Modal definitions (CPU/GPU separation)
└── requirements.txt       # vllm, litellm, dspy, qdrant-client, networkx, grep-ast
```

## 🏗️ Pydantic Data Schemas

### Alt_git Schemas (Workspace 1)
```python
from pydantic import BaseModel, Field
from typing import Dict, List

class FileChange(BaseModel):
    filepath: str
    content: str
    change_type: str = Field(description="'add', 'modify', or 'delete'")

class CommitPayload(BaseModel):
    author: str
    message: str
    changes: List[FileChange]
    issue_id: str

class RepoStateResponse(BaseModel):
    current_hash: str
    files: Dict[str, str] = Field(description="Dictionary mapping filepaths to raw file contents")
```

### dev_fleet State Schemas (Workspace 2)
```python
from pydantic import BaseModel
from typing import List

class CrashReport(BaseModel):
    issue_id: str
    traceback: str
    iteration: int = 1

class AgentStateLog(BaseModel):
    issue_id: str
    ground_truth_summary: str
    previous_actions: List[str]
    current_plan: str
```

## 🚀 Smart Order of Development

### Phase 1: Foundation (Parallel Tracks)
**Track A (Alt_git API):** Build the FastAPI backend, define Pydantic schemas, implement the `/telemetry/crash` webhook.

**Track B (dev_fleet Context):** Deploy `doc_ingestion.py` using requests-cache. Populate the Modal Volume.

### Phase 2: Orchestration (Sequential)
**Step 1:** Create the `dev_fleet` worker with a global try/except block to POST crash tracebacks to `Alt_git`.

**Step 2:** Build the `fleet_fix.orchestrator` using DSPy and LiteLLM to route tracebacks to the Jules API.

### Phase 3: The AI-Native Pivot (Parallel Tracks)
**Track A (Knowledge Graph Refactoring):** Upgrade `doc_ingestion.py` to an AI-native pipeline. Use Qwen/qwen3-embedding to chunk/embed docs into Qdrant. Use Qwen/qwen3-reranker to create highly relevant reference nodes in a NetworkX graph.

**Track B (Local Fleet Deployment):** Deploy local Qwen via vLLM. Configure LiteLLM for Shadow Mode (Jules + Qwen parallel routing).

### Phase 4: Independence (Sequential)
Evaluate local Qwen proposals against Jules's successes. Retire Jules and point LiteLLM exclusively to the Qwen fleet.

## ⚠️ Strict Modal Operational Rules
* **No Ephemeral Apps:** Use `modal deploy`, not `modal run`. This is a headless-first deployment.
* **Sandbox Cleanup:** Programmatically delete/stop prior apps and volumes in `Jules_Test_Lab` before testing new deployments to prevent state contamination.
* **Asynchronous Log Capture:** Do NOT wait for Modal deployment standard output logs. Use the `/telemetry/crash` POST webhook logic.
* **Single Thread Enforcement:** You must operate entirely within the assigned `JULES_SESSION_ID`. Wait for async callbacks.
