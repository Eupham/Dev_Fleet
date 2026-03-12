"""LLM Client — Thin wrapper around Modal-native inference calls.

All orchestrator modules call ``chat_completion()`` which routes to
``Inference().generate.remote()`` — no HTTP overhead, no idle timeouts.
"""

from __future__ import annotations

from typing import Any


def chat_completion(
    messages: list[dict[str, str]],
    model: str = "llm",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Send a chat completion request via Modal-native RPC.

    Returns the generated text from the model.
    """
    from inference.server import Inference

    return Inference().generate.remote(
        messages, model=model, temperature=temperature, max_tokens=max_tokens,
    )


def generate(
    context: str,
    task_description: str,
    model: str = "llm",
) -> str:
    """Query the 32B vLLM inference service for a code/plan response.

    Parameters
    ----------
    context:
        GraphRAG context window built from the Tri-Graph memory.
    task_description:
        The atomic task description.
    model:
        Served model alias.

    Returns
    -------
    The generated text from the model.
    """
    system = (
        "You are a senior software engineer. Use the CONTEXT below to "
        "complete the TASK. Respond with code or a precise plan.\n\n"
        f"CONTEXT:\n{context}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_description},
    ]
    return chat_completion(messages, model=model, temperature=0.3, max_tokens=4096)
