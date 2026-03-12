"""Microservice B — Tri-Graph Orchestrator (Modal App definition).

Hosts the 7B decomposition/reranking model on an L40S GPU and
coordinates graph memory, Frege parsing, reranking, sandbox execution,
and the main agent loop.

Deployment (detached):
    modal deploy orchestrator/core_app.py --detach

Logs:
    modal app logs devfleet-orchestrator
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import modal

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
    .pip_install(
        "networkx>=3.2",
        "pydantic>=2.5",
        "httpx>=0.27",
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
# Modal App
# ---------------------------------------------------------------------------

app = modal.App(
    "devfleet-orchestrator",
    secrets=[modal.Secret.from_name("devfleet-modal-secrets")],
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
# Orchestrator entrypoint — invoked via `modal run` or scheduled
# ---------------------------------------------------------------------------


@app.function(
    image=orchestrator_image,
    volumes={
        "/state": graph_state_vol,
        "/workspace": workspace_vol,
    },
    timeout=30 * 60,
)
def run_agent(user_prompt: str, inference_url: str) -> dict:
    """Execute a single agent loop iteration for *user_prompt*.

    Parameters
    ----------
    user_prompt:
        The natural-language task from the user.
    inference_url:
        Base URL of the deployed vLLM inference service
        (e.g. ``https://…--devfleet-inference-serve.modal.run``).

    Returns
    -------
    dict  — final episodic graph state serialised as JSON-compatible dict.
    """
    from orchestrator.agent_loop import agent_loop  # local import inside container

    log_event("agent_start", {"prompt": user_prompt[:200]})

    result = agent_loop(user_prompt, inference_url)

    log_event("agent_complete", {"nodes": len(result.get("nodes", []))})
    return result


@app.local_entrypoint()
def main(
    prompt: str = "Write a Python function that sorts a list using merge sort.",
    inference_url: str = "",
):
    """CLI entrypoint for quick testing: ``modal run orchestrator/core_app.py``."""
    if not inference_url:
        print(
            "WARNING: --inference-url not provided; the agent will not be "
            "able to reach the 32B model for generation."
        )
    result = run_agent.remote(prompt, inference_url)
    print(json.dumps(result, indent=2))
