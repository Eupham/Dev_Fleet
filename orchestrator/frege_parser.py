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
    description: str = Field(...)
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
    intent_observation: str = Field(
        default="",
        description="A brief, objective analysis of the user's core goal.",
    )
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

DECOMPOSITION_SYSTEM = """You are a task decomposition engine operating inside an isolated sandbox.

First, write an `intent_observation` that captures the user's core goal in one or two sentences.

Then generate the atomic sub-tasks needed to fulfil that goal. Use the Codebase Mini-Map below as your guide — it lists relevant file paths and a hint about each file's contents, but you do NOT have the actual file contents yet. Your first sub-tasks must therefore use bash (`cat`, `grep`, `find`) or Python (`open()`, `pathlib`) to read and explore those files before attempting any modifications. Generate only as many tasks as the complexity warrants."""


def _build_decomposition_messages(
    user_prompt: str,
    codebase_context: str = "",
) -> list[dict[str, str]]:
    context_block = (
        f"Codebase Mini-Map (relevant files — contents not yet loaded):\n{codebase_context}\n\n"
        if codebase_context
        else ""
    )
    explicit_user_message = (
        "The following text is the user prompt:\n"
        f"{user_prompt}\n\n"
        f"{context_block}"
        "Decompose this into atomic sub-tasks in a quantity suitable to its complexity."
    )
    return [
        {"role": "system", "content": DECOMPOSITION_SYSTEM},
        {"role": "user", "content": explicit_user_message},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_prompt(
    user_prompt: str,
    model: str = "llm",
    codebase_context: str = "",
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

    messages = _build_decomposition_messages(user_prompt, codebase_context=codebase_context)

    # We pass the schema directly; the backend uses xgrammar to enforce structured generation.
    # We use a higher max_tokens (4096) to prevent EOF errors when parsing complex logic.
    try:
        dag: TaskDAG = chat_completion(
            messages, model=model, temperature=0.2, max_tokens=4096, schema=TaskDAG
        )
    except Exception as e:
        import logging
        logger = logging.getLogger("dev_fleet.frege_parser")

        # Check if it's an EOF validation error (e.g. from Pydantic JSON decoding)
        if "EOF" in str(e) or "truncated" in str(e).lower():
            logger.warning("Caught EOF/Truncated JSON error during decomposition. Retrying...")
            messages.append({"role": "assistant", "content": "The output was interrupted."})
            messages.append({"role": "user", "content": "Your previous JSON response was truncated. Please output the complete JSON object."})
            try:
                dag: TaskDAG = chat_completion(
                    messages, model=model, temperature=0.2, max_tokens=4096, schema=TaskDAG
                )
            except Exception as e2:
                logger.warning("Retry task decomposition failed (%s). Falling back to single-task DAG.", e2)
                return TaskDAG(
                    user_prompt=user_prompt,
                    tasks=[AtomicTaskNode(description=user_prompt[:500])],
                )
        else:
            # Fallback: treat the entire prompt as a single task so the agent always proceeds.
            logger.warning("Task decomposition failed (%s). Falling back to single-task DAG.", e)
            return TaskDAG(
                user_prompt=user_prompt,
                tasks=[AtomicTaskNode(description=user_prompt[:500])],
            )

    # Populate user_prompt if empty to be safe
    dag.user_prompt = user_prompt

    return dag
