"""Orchestrator — Tri-Graph agent loop (CPU container).

Coordinates graph memory, Frege parsing, Qwen3-Reranker scoring,
sandbox execution, and the main agent loop.  All inference calls
use Modal-native RPC to the Inference and Reranker classes within
the same ``devfleet`` app — no HTTP overhead or idle timeouts.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import modal

from fleet_app import app  # shared app defined in app.py

# ---------------------------------------------------------------------------
# Logging — all critical state goes to stdout for `modal app logs`
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("devfleet.orchestrator")

# ---------------------------------------------------------------------------
# Container image — lightweight CPU image for orchestration logic
# ---------------------------------------------------------------------------

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("inference", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .pip_install(
        "networkx>=3.2",
        "pydantic>=2.5",
    )
)

# ---------------------------------------------------------------------------
# Volumes / state persistence
# ---------------------------------------------------------------------------

graph_state_vol = modal.Volume.from_name(
    "devfleet-graph-state", create_if_missing=True
)
workspace_vol = modal.Volume.from_name(
    "devfleet-workspace", create_if_missing=True
)


# ---------------------------------------------------------------------------
# Structured logging helper
# ---------------------------------------------------------------------------


def log_event(event_type: str, payload: dict | None = None) -> None:
    """Emit a structured JSON log line for ``modal app logs`` consumption."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **(payload or {}),
    }
    logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Orchestrator entrypoint — invoked via `modal run app.py`
# ---------------------------------------------------------------------------


@app.function(
    image=orchestrator_image,
    volumes={
        "/state": graph_state_vol,
        "/workspace": workspace_vol,
    },
    timeout=30 * 60,
    retries=0,
)
def run_agent(user_prompt: str) -> dict:
    """Execute a single agent loop iteration for *user_prompt*.

    Parameters
    ----------
    user_prompt:
        The natural-language task from the user.

    Returns
    -------
    dict  — final episodic graph state serialised as JSON-compatible dict.
    """
    from orchestrator.agent_loop import agent_loop  # local import inside container

    log_event("agent_start", {"prompt": user_prompt[:200]})

    result = agent_loop(user_prompt)

    log_event("agent_complete", {"nodes": len(result.get("nodes", []))})
    return result
