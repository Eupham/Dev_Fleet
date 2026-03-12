"""Frege Parser — Decomposes user prompts into a Task DAG.

Uses Frege's compositionality principle: a complex prompt is structurally
decomposed into a Directed Acyclic Graph of *atomic* tasks.  Each node is
validated via a strict Pydantic schema.

The decomposition is performed by the 7B model (hosted locally on the
orchestrator GPU or queried via the inference URL).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AtomicTaskNode(BaseModel):
    """A single indivisible task produced by Frege decomposition."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = Field(..., max_length=500)
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of tasks this node depends on (DAG edges).",
    )
    tool_hint: str = Field(
        default="",
        description="Optional hint: 'bash', 'python', 'git', or empty.",
    )
    status: str = Field(
        default="pending",
        description="pending | running | success | failed",
    )


class TaskDAG(BaseModel):
    """The complete decomposition of a user prompt."""

    user_prompt: str
    tasks: list[AtomicTaskNode]


# ---------------------------------------------------------------------------
# Decomposition prompt template
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM = """You are a task decomposition engine.
Given a user prompt, break it into the SMALLEST possible atomic sub-tasks.
Return ONLY a JSON array of objects with keys: "description", "depends_on" (list of indices, 0-based), "tool_hint".
Do NOT include any explanation outside the JSON array."""


def _build_decomposition_messages(
    user_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DECOMPOSITION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_prompt(
    user_prompt: str,
    inference_url: str,
    model: str = "llm",
    timeout: float = 120.0,
) -> TaskDAG:
    """Decompose *user_prompt* into a ``TaskDAG`` via the LLM.

    Parameters
    ----------
    user_prompt:
        The raw natural-language request.
    inference_url:
        Base URL of the vLLM inference service.
    model:
        Served model name (default ``"llm"``).
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    TaskDAG validated by Pydantic.
    """
    messages = _build_decomposition_messages(user_prompt)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    resp = httpx.post(
        f"{inference_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]

    # The model may wrap JSON in markdown fences — strip them.
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    raw_tasks: list[dict[str, Any]] = json.loads(content)

    # Build AtomicTaskNode list, mapping index-based depends_on → UUIDs
    nodes: list[AtomicTaskNode] = []
    for item in raw_tasks:
        nodes.append(
            AtomicTaskNode(
                description=item.get("description", ""),
                tool_hint=item.get("tool_hint", ""),
            )
        )

    # Resolve depends_on indices to actual node IDs
    for i, item in enumerate(raw_tasks):
        deps = item.get("depends_on", [])
        nodes[i].depends_on = [nodes[int(d)].id for d in deps if int(d) < len(nodes)]

    return TaskDAG(user_prompt=user_prompt, tasks=nodes)
