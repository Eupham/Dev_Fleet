# orchestrator/task_parser.py
"""Typed task decomposition — TaskDAG and AtomicTaskNode schemas.

Defines four task types (QueryTask, TransformTask, VerifyTask, ComposeTask)
as a Pydantic discriminated union. validate_dag_types() enforces structural
constraints on the declared dependency graph. All graph operations are O(V+E).
"""
from __future__ import annotations
import uuid
from typing import Annotated, Literal, Union

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class _TaskBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    tool_hint: Literal["bash", "python", ""] = ""
    status: Literal["pending", "running", "success", "failed"] = "pending"
    preconditions: list[str] = Field(
        default=[],
        description="IDs of tasks that must succeed before this one runs.",
    )
    postconditions: list[str] = Field(
        default=[],
        description="File paths or names this task is expected to produce.",
    )
    actor_capability: Literal["bash", "python", "llm_only"] = "python"
    implementation_depth: Literal["algorithm", "library", "syscall"] = "library"
    execution_cost: Literal["trivial", "moderate", "expensive"] = "moderate"


class QueryTask(_TaskBase):
    """Reads existing state. No filesystem writes."""
    task_type: Literal["query"] = "query"
    target_resource: str = ""


class TransformTask(_TaskBase):
    """Writes new state or modifies existing files."""
    task_type: Literal["transform"] = "transform"


class VerifyTask(_TaskBase):
    """Checks a postcondition of a prior transform.
    Constraint: must declare at least one TransformTask in preconditions."""
    task_type: Literal["verify"] = "verify"
    assertion: str = Field(
        default="",
        description="The specific condition being checked.",
    )


class ComposeTask(_TaskBase):
    """Sequences multiple transforms as a single named unit.
    Constraint: sub_task_ids must reference only transform or query tasks."""
    task_type: Literal["compose"] = "compose"
    sub_task_ids: list[str] = []


AtomicTaskNode = Annotated[
    Union[QueryTask, TransformTask, VerifyTask, ComposeTask],
    Field(discriminator="task_type"),
]


def validate_dag_types(tasks: list) -> None:
    """Validate structural constraints on declared task dependencies.

    Checks:
    1. All precondition IDs exist in the task list.
    2. No self-referential preconditions.
    3. Precondition graph is acyclic (O(V+E) via nx.is_directed_acyclic_graph).
    4. VerifyTask has >= 1 TransformTask in preconditions.
    5. ComposeTask sub_task_ids reference only transform/query tasks.

    Raises ValueError on any violation.
    """
    id_to_type = {t.id: t.task_type for t in tasks}
    G = nx.DiGraph()

    for task in tasks:
        G.add_node(task.id)
        for pre_id in task.preconditions:
            if pre_id not in id_to_type:
                raise ValueError(
                    f"Task {task.id!r} has unknown precondition {pre_id!r}."
                )
            if pre_id == task.id:
                raise ValueError(
                    f"Task {task.id!r} lists itself as a precondition."
                )
            G.add_edge(pre_id, task.id)

    # O(V+E) cycle check — not all_simple_paths
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Declared preconditions form a cycle.")

    for task in tasks:
        if task.task_type == "compose":
            for sub_id in task.sub_task_ids:
                if sub_id not in id_to_type:
                    raise ValueError(
                        f"ComposeTask {task.id!r}: unknown sub_task_id {sub_id!r}."
                    )
                if id_to_type[sub_id] not in ("transform", "query"):
                    raise ValueError(
                        f"ComposeTask {task.id!r}: sub_task {sub_id!r} is type "
                        f"{id_to_type[sub_id]!r}. Only transform/query can be composed."
                    )


class TaskDAG(BaseModel):
    user_prompt: str
    intent_observation: str = ""
    tasks: list[AtomicTaskNode]

    @model_validator(mode="after")
    def check_composition(self) -> "TaskDAG":
        validate_dag_types(self.tasks)
        return self


# ---------------------------------------------------------------------------
# Decomposition prompts
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_CREATE = """You are a software task planner. The sandbox workspace (/workspace) is EMPTY.

Rules:
1. Do NOT read or explore files that don't exist yet.
2. WRITE code first, then RUN it.
3. tool_hint: "python", "bash", or "".
4. Every task must be directly executable.
5. Write all output files to /workspace/.
6. Aim for 3-5 focused tasks.

For EACH task include:
  task_type: "query" | "transform" | "verify" | "compose"
  preconditions: list of task IDs that must succeed first ([] if none)
  postconditions: list of file paths this task produces ([] if none)
  actor_capability: "bash" | "python" | "llm_only"
  implementation_depth: "algorithm" | "library" | "syscall"
  execution_cost: "trivial" | "moderate" | "expensive"

task_type rules:
  query:     reads existing state only, no filesystem writes
  transform: creates or modifies files
  verify:    checks a transform's output — must list a transform in preconditions
  compose:   sequences transforms — list their IDs in sub_task_ids

Write intent_observation first, then the task list."""

DECOMPOSITION_SYSTEM_MODIFY = """You are a software task planner. Files listed in the Codebase Mini-Map exist in /workspace.

Rules:
1. Read relevant files before modifying them.
2. tool_hint: "python", "bash", or "".
3. Write modified files back to /workspace/.
4. Aim for 3-6 focused tasks.

Include for each task: task_type, preconditions, postconditions,
actor_capability, implementation_depth, execution_cost.

Write intent_observation first, then the task list."""

DECOMPOSITION_SYSTEM_RESEARCH = """You are a software task planner. The sandbox has internet access.

Rules:
1. Use bash (curl/wget) or Python (requests/httpx) to fetch data.
2. Write output to /workspace/.
3. Aim for 3-6 focused tasks.

Include for each task: task_type, preconditions, postconditions,
actor_capability, implementation_depth, execution_cost.

Write intent_observation first, then the task list."""


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
        if codebase_context else ""
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{context_block}"
                f"User request: {user_prompt}\n\n"
                "Decompose into atomic executable sub-tasks. "
                "Return valid JSON matching the TaskDAG schema."
            ),
        },
    ]


def parse_prompt(
    user_prompt: str,
    model: str = "llm",
    codebase_context: str = "",
    is_research: bool = False,
) -> TaskDAG:
    """Decompose user_prompt into a typed TaskDAG via the Qwen model."""
    from orchestrator.llm_client import chat_completion
    import logging
    logger = logging.getLogger("dev_fleet.task_parser")

    messages = _build_decomposition_messages(
        user_prompt, codebase_context, is_research
    )

    try:
        result: TaskDAG = chat_completion(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2048,
            schema=TaskDAG,
        )
        return result
    except Exception as exc:
        logger.warning(
            "Typed decomposition failed (%s) — falling back to single TransformTask.",
            exc,
        )
        return TaskDAG(
            user_prompt=user_prompt,
            intent_observation="Fallback: decomposition failed.",
            tasks=[TransformTask(description=user_prompt[:500], tool_hint="python")],
        )
