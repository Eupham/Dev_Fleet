"""Dev Fleet — Unified Modal app entry point.

Single app with GPU inference, a lightweight reranker, and CPU
orchestration running as separate containers within the same
Modal application.

Deployment:
    modal deploy app.py

Logs:
    modal app logs dev_fleet
"""

import json

import modal

from fleet_app import app

# Register all functions / classes by importing their modules.
# Each module does ``from app import app`` and decorates its own
# functions/classes with the shared ``app`` object.
import inference.server  # noqa: E402 — registers Inference (moderate/complex tier, L40S)
import inference.reranker  # noqa: E402 — registers Reranker class
import inference.embedder  # noqa: E402 — registers Embedder class
import inference.model_pool  # noqa: E402 — registers InferenceSmall/Medium/Large
import orchestrator.core_app  # noqa: E402 — registers run_agent
import ui.web  # noqa: E402 — registers UI web endpoint


@app.local_entrypoint()
def main(
    prompt: str = "Write a Python function that sorts a list using merge sort.",
):
    """CLI entrypoint for quick testing: ``modal run app.py``."""
    from orchestrator.core_app import run_agent

    result = run_agent.remote(prompt)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
async def warmup_fleet():
    """Trigger GPU snapshots for all inference tiers in parallel after deployment."""
    import asyncio
    from inference.server import Inference
    from inference.model_pool import InferenceSmall, InferenceMedium, InferenceLarge

    print("🚀 Triggering parallel GPU snapshots for the entire fleet...")
    print("⏳ This will take up to 30 minutes. You can monitor progress in the Modal dashboard.")
    
    dummy_msg = [{"role": "user", "content": "Hello!"}]
    
    # Fire all generation requests concurrently so Modal boots all 4 GPUs at once
    results = await asyncio.gather(
        Inference().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceSmall().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceMedium().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceLarge().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        return_exceptions=True  # Prevent one timeout from cancelling the others
    )

    for i, res in enumerate(["Moderate (L40S)", "Trivial (T4)", "Simple (A10G)", "Expert (A100)"]):
        if isinstance(results[i], Exception):
            print(f"⚠️ {res} snapshot failed or timed out: {results[i]}")
        else:
            print(f"✅ {res} snapshot successfully captured!")


@app.local_entrypoint()
async def warmup_fleet():
    """Trigger GPU snapshots for all inference tiers in parallel after deployment."""
    import asyncio
    from inference.server import Inference
    from inference.model_pool import InferenceSmall, InferenceMedium, InferenceLarge

    print("🚀 Triggering parallel GPU snapshots for the entire fleet...")
    print("⏳ This will take up to 30 minutes. You can monitor progress in the Modal dashboard.")
    
    dummy_msg = [{"role": "user", "content": "Hello!"}]
    
    # Fire all generation requests concurrently so Modal boots all 4 GPUs at once
    results = await asyncio.gather(
        Inference().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceSmall().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceMedium().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceLarge().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        return_exceptions=True  # Prevent one timeout from cancelling the others
    )

    for i, res in enumerate(["Moderate (L40S)", "Trivial (T4)", "Simple (A10G)", "Expert (A100)"]):
        if isinstance(results[i], Exception):
            print(f"⚠️ {res} snapshot failed or timed out: {results[i]}")
        else:
            print(f"✅ {res} snapshot successfully captured!")


@app.local_entrypoint()
async def warmup_fleet():
    """Trigger GPU snapshots for all inference tiers in parallel after deployment."""
    import asyncio
    from inference.server import Inference
    from inference.model_pool import InferenceSmall, InferenceMedium, InferenceLarge

    print("🚀 Triggering parallel GPU snapshots for the entire fleet...")
    print("⏳ This will take up to 30 minutes. You can monitor progress in the Modal dashboard.")
    
    dummy_msg = [{"role": "user", "content": "Hello!"}]
    
    # Fire all generation requests concurrently so Modal boots all 4 GPUs at once
    results = await asyncio.gather(
        Inference().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceSmall().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceMedium().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        InferenceLarge().generate.remote.aio(messages=dummy_msg, max_tokens=1),
        return_exceptions=True  # Prevent one timeout from cancelling the others
    )

    for i, res in enumerate(["Moderate (L40S)", "Trivial (T4)", "Simple (A10G)", "Expert (A100)"]):
        if isinstance(results[i], Exception):
            print(f"⚠️ {res} snapshot failed or timed out: {results[i]}")
        else:
            print(f"✅ {res} snapshot successfully captured!")
