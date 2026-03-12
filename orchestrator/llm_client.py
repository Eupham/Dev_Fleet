"""LLM Client — Sends generation requests to the 32B vLLM service."""

from __future__ import annotations

from typing import Any

import httpx


def generate(
    context: str,
    task_description: str,
    inference_url: str,
    model: str = "llm",
) -> str:
    """Query the 32B vLLM inference service for a code/plan response.

    Parameters
    ----------
    context:
        GraphRAG context window built from the Tri-Graph memory.
    task_description:
        The atomic task description.
    inference_url:
        Base URL of the deployed vLLM inference service.
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
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_description},
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    resp = httpx.post(
        f"{inference_url}/v1/chat/completions",
        json=payload,
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
