"""Typed task decomposition — TaskDAG and AtomicTaskNode schemas.

Implements Montague Grammar (translation) and Fillmore Frame Semantics (instantiation).
"""
from __future__ import annotations
import uuid
from typing import Annotated, Literal, Union

import networkx as nx
from pydantic import BaseModel, Field, model_validator

# --- 1. FILLMORE FRAMES (Core Task Schemas) ---

class _TaskBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    tool_hint: Literal["bash", "python", ""] = "python"
    status: Literal["pending", "running", "success", "failed"] = "pending"
    preconditions: list[str] = Field(default=[])
    postconditions: list[str] = Field(default=[])
    actor_capability: Literal["bash", "python", "llm_only"] = "python"
    implementation_depth: Literal["algorithm", "library", "syscall"] = "library"
    execution_cost: Literal["trivial", "moderate", "expensive"] = "moderate"

class QueryTask(_TaskBase):
    task_type: Literal["query"] = "query"
    target_resource: str = ""

class TransformTask(_TaskBase):
    task_type: Literal["transform"] = "transform"

class VerifyTask(_TaskBase):
    task_type: Literal["verify"] = "verify"
    assertion: str = ""

AtomicTaskNode = Annotated[
    Union[QueryTask, TransformTask, VerifyTask],
    Field(discriminator="task_type"),
]

class TaskDAG(BaseModel):
    user_prompt: str = ""
    intent_observation: str = ""
    tasks: list[AtomicTaskNode]

# --- 2. MONTAGUE TRANSLATION LAYER (LLM Interface) ---

class FlatMontagueParse(BaseModel):
    """Safe, flat schema for llama.cpp GBNF compiler. Represents a single logical action."""
    action_verb: Literal["create", "modify", "read", "verify", "research"] = Field(
        ..., description="The fundamental action type."
    )
    theme_target: str = Field(
        default="", description="The file or resource target. E.g. '/workspace/app.py'"
    )
    goal_instruction: str = Field(
        ..., description="The formal instruction of what needs to be accomplished."
    )

class MontagueDecomposition(BaseModel):
    """The root schema passed to the LLM."""
    intent_observation: str = Field(default="Translating request to formal logic.")
    parses: list[FlatMontagueParse] = Field(..., description="Sequential list of actions.")

def map_to_fillmore_frames(parses: list[FlatMontagueParse]) -> list[AtomicTaskNode]:
    """Deterministically map flat linguistic parses into strict Fillmore frames."""
    frames = []
    prev_id = None
    for p in parses:
        if p.action_verb in ("create", "modify", "research"):
            frame = TransformTask(
                id=uuid.uuid4().hex[:8],
                description=f"Target: {p.theme_target}\nGoal: {p.goal_instruction}",
                tool_hint="python" if p.action_verb != "research" else "bash"
            )
        elif p.action_verb == "read":
            frame = QueryTask(
                id=uuid.uuid4().hex[:8],
                description=p.goal_instruction,
                target_resource=p.theme_target
            )
        elif p.action_verb == "verify":
            frame = VerifyTask(
                id=uuid.uuid4().hex[:8],
                description=p.goal_instruction,
                assertion=p.theme_target
            )
        else:
            frame = TransformTask(id=uuid.uuid4().hex[:8], description=p.goal_instruction)

        if prev_id: # Chain sequentially
            frame.preconditions = [prev_id]
        frames.append(frame)
        prev_id = frame.id
    return frames

# --- 3. PIPELINE ---

def parse_prompt(user_prompt: str, model: str = "llm", codebase_context: str = "", is_research: bool = False) -> TaskDAG:
    from orchestrator.llm_client import chat_completion
    import logging
    logger = logging.getLogger("dev_fleet.task_parser")

    messages = [
        {"role": "system", "content": "You are a Montague Parser. Break the request into flat sequential actions. Return JSON."},
        {"role": "user", "content": f"Context:\n{codebase_context}\n\nRequest: {user_prompt}"}
    ]

    try:
        raw_response = chat_completion(messages, model=model, schema=MontagueDecomposition)
        
        if isinstance(raw_response, MontagueDecomposition):
            parsed = raw_response
        elif isinstance(raw_response, dict):
            parsed = MontagueDecomposition.model_validate(raw_response)
        else:
            raise ValueError("Unexpected response type")

        typed_tasks = map_to_fillmore_frames(parsed.parses)
        return TaskDAG(user_prompt=user_prompt, intent_observation=parsed.intent_observation, tasks=typed_tasks)

    except Exception as exc:
        logger.warning("Montague decomposition failed (%s) — falling back.", exc)
        return TaskDAG(
            user_prompt=user_prompt,
            intent_observation="Fallback: single task.",
            tasks=[TransformTask(description=user_prompt[:500], tool_hint="python")]
        )
