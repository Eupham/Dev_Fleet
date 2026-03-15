"""Compositional Task Parser — Decomposes user prompts into a Task DAG.

Implements Frege's compositionality principle operationally: the meaning of
the whole task is a function of the meanings of its atomic parts and their
combination rules.

Composition structure is derived from execution (see orchestrator/composition.py),
not declared by the model. AtomicTaskNode is intentionally minimal — descriptions
and tool hints only. Dependency edges are inferred post-execution by observing
filesystem transitions.

Decomposition is performed by the instruction-tuned Qwen model via Modal-native RPC.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AtomicTaskNode(BaseModel):
    """A single indivisible task produced by compositional decomposition.

    Intentionally minimal: the model provides descriptions and tool hints.
    All composition structure (dependencies, I/O contracts) is derived from
    execution by observing filesystem transitions — not declared by the model.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = Field(...)
    tool_hint: str = Field(
        default="",
        description="Optional hint: 'bash', 'python', or empty.",
    )
    status: str = Field(
        default="pending",
        description="pending | running | success | failed",
    )


class TaskDAG(BaseModel):
    """The complete compositional decomposition of a user prompt.

    A simple container: the LLM's suggested task ordering is a heuristic.
    The runtime may override it based on observed composition structure from
    CompositionLedger after execution.
    """

    user_prompt: str
    intent_observation: str = Field(
        default="",
        description="A brief, objective analysis of the user's core goal.",
    )
    tasks: list[AtomicTaskNode]


# ---------------------------------------------------------------------------
# Decomposition prompt template
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_CREATE = """You are a software task planner. The sandbox workspace (/workspace) is EMPTY.

Rules:
1. Do NOT create tasks that read or explore files that don't exist yet.
2. WRITE code first, then RUN it. Never read before writing when creating from scratch.
3. tool_hint: use "python" to run Python code, "bash" to run shell commands, "" for planning only.
4. Every task must be directly executable — no placeholders or "verify" steps.
5. Write all output files to /workspace/.
6. Aim for 3-5 focused tasks. Do not over-decompose.

First write intent_observation (one sentence), then list the tasks."""

DECOMPOSITION_SYSTEM_MODIFY = """You are a software task planner. The Codebase Mini-Map lists files already in /workspace.

Rules:
1. Read relevant files first (bash cat/grep or Python open) before modifying them.
2. tool_hint: "python", "bash", or "".
3. Write modified files back to /workspace/.
4. Aim for 3-6 focused tasks.

First write intent_observation, then list the tasks."""

DECOMPOSITION_SYSTEM_RESEARCH = """You are a software task planner. The sandbox has internet access.

Rules:
1. Use bash (curl/wget) or Python (requests/httpx) to fetch data from the web.
2. Write output and any created tools to /workspace/.
3. tool_hint: "python", "bash", or "".
4. Aim for 3-6 focused tasks.

First write intent_observation, then list the tasks."""


def _build_decomposition_messages(
    user_prompt: str,
    codebase_context: str = "",
    is_research: bool = False,
) -> list[dict[str, str]]:
    if is_research:
        system = DECOMPOSITION_SYSTEM_RESEARCH
    elif codebase_context:
        system = DECOMPOSITION_SYSTEM_MODIFY
    else:
        system = DECOMPOSITION_SYSTEM_CREATE

    context_block = (
        f"Codebase Mini-Map (files already in /workspace):\n{codebase_context}\n\n"
        if codebase_context
        else ""
    )
    user_message = (
        f"{context_block}"
        f"User request: {user_prompt}\n\n"
        "Decompose into atomic executable sub-tasks."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_prompt(
    user_prompt: str,
    model: str = "llm",
    codebase_context: str = "",
    is_research: bool = False,
) -> TaskDAG:
    """Decompose *user_prompt* into a ``TaskDAG`` via the Qwen model.

    Parameters
    ----------
    user_prompt:
        The raw natural-language request.
    model:
        Served model name (default ``"llm"``).
    codebase_context:
        Optional codebase mini-map from RAG retrieval.
    is_research:
        Whether this is a research-oriented task (use research prompt).

    Returns
    -------
    TaskDAG validated by Pydantic.
    """
    from orchestrator.llm_client import chat_completion

    messages = _build_decomposition_messages(
        user_prompt, codebase_context=codebase_context, is_research=is_research
    )

    try:
        dag: TaskDAG = chat_completion(
            messages, model=model, temperature=0.2, max_tokens=4096, schema=TaskDAG
        )
    except Exception as e:
        import logging
        logger = logging.getLogger("dev_fleet.frege_parser")

        if "EOF" in str(e) or "truncated" in str(e).lower():
            logger.warning("Caught EOF/Truncated JSON error during decomposition. Retrying...")
            messages.append({"role": "assistant", "content": "The output was interrupted."})
            messages.append({"role": "user", "content": "Your previous JSON response was truncated. Please output the complete JSON object."})
            try:
                dag = chat_completion(
                    messages, model=model, temperature=0.2, max_tokens=4096, schema=TaskDAG
                )
            except Exception as e2:
                logger.warning("Retry decomposition failed (%s). Falling back to single-task DAG.", e2)
                return TaskDAG(
                    user_prompt=user_prompt,
                    tasks=[AtomicTaskNode(description=user_prompt[:500])],
                )
        else:
            logger.warning("Task decomposition failed (%s). Falling back to single-task DAG.", e)
            return TaskDAG(
                user_prompt=user_prompt,
                tasks=[AtomicTaskNode(description=user_prompt[:500])],
            )

    dag.user_prompt = user_prompt
    return dag
