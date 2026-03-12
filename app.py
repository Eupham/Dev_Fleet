"""Dev Fleet — Unified Modal app entry point.

Single app with GPU inference, a lightweight reranker, and CPU
orchestration running as separate containers within the same
Modal application.

Deployment:
    modal deploy app.py

Logs:
    modal app logs devfleet
"""

import json

import modal

app = modal.App("devfleet")

# Register all functions / classes by importing their modules.
# Each module does ``from app import app`` and decorates its own
# functions/classes with the shared ``app`` object.
import inference.server  # noqa: E402 — registers Inference class
import inference.reranker  # noqa: E402 — registers Reranker class
import orchestrator.core_app  # noqa: E402 — registers run_agent


@app.local_entrypoint()
def main(
    prompt: str = "Write a Python function that sorts a list using merge sort.",
):
    """CLI entrypoint for quick testing: ``modal run app.py``."""
    from orchestrator.core_app import run_agent

    result = run_agent.remote(prompt)
    print(json.dumps(result, indent=2))
