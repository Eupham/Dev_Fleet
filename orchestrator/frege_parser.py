"""Frege Parser — Decomposes user prompts into a Task DAG.

Uses Frege's compositionality principle: a complex prompt is structurally
decomposed into a Directed Acyclic Graph of *atomic* tasks.  Each node is
validated via a strict Pydantic schema.

Decomposition is performed by the 32B model via Modal-native RPC.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

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
Given a user prompt, break it into the SMALLEST possible atomic sub-tasks."""


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
    model: str = "llm",
) -> TaskDAG:
    """Decompose *user_prompt* into a ``TaskDAG`` via the 32B model.

    Parameters
    ----------
    user_prompt:
        The raw natural-language request.
    model:
        Served model name (default ``"llm"``).

    Returns
    -------
    TaskDAG validated by Pydantic.
    """
    from orchestrator.llm_client import chat_completion

    messages = _build_decomposition_messages(user_prompt)

    # We pass the schema directly; the backend uses outlines to enforce structured generation
    dag: TaskDAG = chat_completion(
        messages, model=model, temperature=0.2, max_tokens=2048, schema=TaskDAG
    )

    # The output is directly guaranteed to be a TaskDAG.
    # Let's populate user_prompt if empty to be safe
    dag.user_prompt = user_prompt

    # Map any index-based depends_on to UUIDs for safety
    nodes = dag.tasks
    for i, item in enumerate(nodes):
        deps = item.depends_on
        # Assuming if depends_on are indices represented as strings, we fix them:
        new_deps = []
        for d in deps:
            if d.isdigit() and int(d) < len(nodes):
                new_deps.append(nodes[int(d)].id)
            else:
                new_deps.append(d)
        nodes[i].depends_on = new_deps

    return dag
