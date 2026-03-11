# Project Initiative: `dev_fleet` & `Alt_git` Migration

## 🎯 Overarching Objective
Transition away from standard GitHub infrastructure by building a fully isolated, headless deployment on Modal. 

The primary workspace (`dev_fleet`) hosts the main application and a dedicated AI maintenance toolset called `fleet_fix` (ff). The secondary workspace (`Alt_git`) operates strictly as a custom, lightweight version control backend. A third workspace acts as an isolated testing sandbox. The ultimate goal is to establish a self-healing codebase managed entirely by a local fleet of Qwen models, bypassing traditional git entirely.

## 🧭 The Core Philosophy: "No Blind Coding"
Both Jules and the `fleet_fix` Qwen models are strictly prohibited from guessing or coding blind. Every debugging or feature loop must strictly adhere to the following sequence:
1. **Research First:** Read offline documentation from the Modal Volume or online docs for any framework or error encountered.
2. **Establish Ground Truth:** Synthesize findings into a `docs/issues/{issue_id}_ground_truth.md` file. This acts as the absolute factual baseline.
3. **Maintain the Diary:** Before executing a fix, log the attempt, the reasoning, and the outcome in `docs/issues/{issue_id}_change_log_diary.md`.

## 🏗️ Architecture & Workspaces
To enforce strict Separation of Concerns (SoC) and safe iteration, responsibilities are divided across three Modal workspaces:

* **Workspace 1: `Alt_git` (Version Control API - Production)**
    * **Function:** Stores the raw codebase, processes file hashes, generates diffs, and executes commits. It tracks *what* changed.
* **Workspace 2: `dev_fleet` (App, Agents, & State - Production)**
    * **Function:** Runs the core application and catches tracebacks. Manages Qdrant vector memory, NetworkX routing, Ground Truth files, and Change Log diaries. It tracks the *context* of the bugs.
* **Workspace 3: `Jules_Test_Lab` (Sandbox)**
    * **Function:** Ephemeral testing ground. Jules deploys here to run and test code changes. Only when a test runs completely clean does Jules commit the code to `Alt_git`.

### 🔑 Secret Management
You have been provisioned with the following exact environment variables. Use them to authenticate your Modal API calls appropriately based on the target workspace:
* `DF_MODAL_TOKEN_ID` & `DF_MODAL_TOKEN_SECRET`: For Workspace 2 (`dev_fleet`)
* `AG_MODAL_TOKEN_ID` & `AG_MODAL_TOKEN_SECRET`: For Workspace 1 (`Alt_git`)
* `Jules_MODAL_TOKEN_ID` & `Jules_MODAL_TOKEN_SECRET`: For Workspace 3 (`Jules_Test_Lab`)
* `JULES_API_KEY`: For API routing via LiteLLM.

---

## ⚠️ Strict Modal Operational Rules

1. **No Ephemeral Apps:** Do NOT use `modal run` or ephemeral app states. All deployments must be persistent using `modal deploy`.
2. **Sandbox Cleanup:** Before initializing a new deployment in `Jules_Test_Lab`, you MUST programmatically delete/stop any prior apps and Modal Volumes in that workspace to prevent state contamination and resource limits.
3. **Asynchronous Log Capture:** Do NOT wait for Modal deployment logs or standard output. It will cause timeouts. 
    * **The Solution:** Wrap all entry points in `dev_fleet` and `Jules_Test_Lab` in a global `try/except` block. 
    * On failure, the app must instantly format the traceback into a JSON payload and HTTP POST it to a dedicated `/telemetry/crash` endpoint hosted on `Alt_git`. 
    * Jules will then execute a fast HTTP GET to `Alt_git` to read the execution results without hanging.

---

## 📚 Core Libraries (Headless First)
No UI templates will be used in this phase. 

**Workspace 1: `Alt_git`**
* **FastAPI** & **Pydantic**
* **DeepDiff** (Complex dict diffing)
* `hashlib` & `difflib` (Standard library)

**Workspace 2 & 3: `dev_fleet` / `Jules_Test_Lab`**
* **vLLM:** High-throughput Qwen Serving
* **LiteLLM:** API routing
* **DSPy:** Neurosymbolic prompt pipelines
* **grep-ast:** Code structural parsing
* **LibCST:** Safe Python code refactoring
* **Qdrant** & **NetworkX**
* **markdownify** & **beautifulsoup4**: For document ingestion

---

## 🏗️ Pydantic Data Schemas

### `Alt_git` Schemas (Workspace 1)
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

### `state` Schemas (Workspace 2 & 3)
from pydantic import BaseModel, Field
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
