"""Frege Parser — Decomposes user prompts into a Task DAG.

Uses Frege's compositionality principle: a complex prompt is structurally
decomposed into a Directed Acyclic Graph of *atomic* tasks.  Each node is
validated via a strict Pydantic schema.

Decomposition is performed by the 32B model via Modal-native RPC.
"""

from __future__ import annotations

import uuid
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskDAG":
        """Resolve index-based depends_on to UUIDs, assert DAG acyclicity, and
        reorder the tasks array via topological sort so that every dependency is
        guaranteed to appear before the task that depends on it."""
        import logging

        logger = logging.getLogger("dev_fleet.frege_parser")

        # Build a dual map: integer index strings and real IDs both resolve to the task ID.
        id_map: dict[str, str] = {str(i): task.id for i, task in enumerate(self.tasks)}
        id_map.update({task.id: task.id for task in self.tasks})

        task_map: dict[str, AtomicTaskNode] = {task.id: task for task in self.tasks}

        G = nx.DiGraph()
        for task in self.tasks:
            resolved: list[str] = []
            for dep in task.depends_on:
                if dep in id_map:
                    resolved.append(id_map[dep])
                else:
                    logger.warning("Dropping unknown dependency %r from task %s", dep, task.id)
            task.depends_on = resolved
            G.add_node(task.id)
            for dep in resolved:
                G.add_edge(dep, task.id)  # directed edge: dependency → dependent

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Task decomposition resulted in a circular dependency.")

        # Reorder tasks so the execution array always satisfies dependency order,
        # regardless of the order the LLM originally emitted the tasks.
        sorted_ids = list(nx.topological_sort(G))
        self.tasks = [task_map[node_id] for node_id in sorted_ids]

        return self


# ---------------------------------------------------------------------------
# Decomposition prompt template
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM = """You are a task decomposition engine.
Given a user prompt, break it into AT MOST 5 atomic sub-tasks.
Be concise — keep each task description under 60 words.
Return the minimal number of tasks needed; simple prompts need only 1-2 tasks."""


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

    # We pass the schema directly; the backend uses xgrammar to enforce structured generation.
    # max_tokens=512 is enough for 5 tasks; the 0.5B model can over-generate without this cap.
    try:
        dag: TaskDAG = chat_completion(
            messages, model=model, temperature=0.2, max_tokens=512, schema=TaskDAG
        )
    except Exception as e:
        # Fallback: treat the entire prompt as a single task so the agent always proceeds.
        import logging
        logging.getLogger("dev_fleet.frege_parser").warning(
            "Task decomposition failed (%s). Falling back to single-task DAG.", e
        )
        return TaskDAG(
            user_prompt=user_prompt,
            tasks=[AtomicTaskNode(description=user_prompt[:500])],
        )

    # Populate user_prompt if empty to be safe
    dag.user_prompt = user_prompt

    return dag
