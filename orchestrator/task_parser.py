"""Typed task decomposition — Montague Grammar and Fillmore Frames.

Implements the two-step linguistic pipeline:
1. Montague Translation: NL -> Flat Logical Actions (LLM)
2. Fillmore Frame Mapping: Actions -> Strict Task Graph (Python)
"""
from __future__ import annotations
import uuid
import logging
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("dev_fleet.task_parser")

# --- 1. FILLMORE FRAMES (Execution Schemas) ---

class _TaskBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    tool_hint: Literal["bash", "python", ""] = "python"
    status: Literal["pending", "running", "success", "failed"] = "pending"
    preconditions: list[str] = Field(default=[])
    postconditions: list[str] = Field(default=[])

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

class MontagueAction(BaseModel):
    """A single flat linguistic parse (Montague Translation)."""
    verb: Literal["create", "modify", "read", "verify", "research"]
    target: str = Field(description="The physical entity (file/URL).")
    instruction: str = Field(description="Actionable instruction.")

class MontagueDecomposition(BaseModel):
    """The root schema passed to the LLM."""
    intent_observation: str = Field(default="Translating request to formal logic.")
    parses: list[MontagueAction] = Field(..., description="Sequential list of actions.")

def parse_prompt(user_prompt: str, model: str = "llm", codebase_context: str = "") -> TaskDAG:
    """The Two-Step Neurosymbolic Pipeline."""
    from orchestrator.llm_client import chat_completion
    
    messages = [
        {"role": "system", "content": "You are a Montague Parser. Extract flat sequential actions into JSON."},
        {"role": "user", "content": f"Context:\n{codebase_context}\n\nRequest: {user_prompt}"}
    ]

    try:
        raw_response = chat_completion(messages, model=model, schema=MontagueDecomposition)
        
        # BULLETPROOF RE-INSTANTIATION: Fixes the 'Zombie Object' AttributeError
        if isinstance(raw_response, MontagueDecomposition):
            # Extract data from the remote object and rebuild locally
            parsed = MontagueDecomposition.model_validate(raw_response.model_dump())
        elif isinstance(raw_response, dict):
            parsed = MontagueDecomposition.model_validate(raw_response)
        else:
            raise ValueError(f"Unexpected response type from LLM: {type(raw_response)}")

        # Python handles the Fillmore mapping and DAG construction
        tasks = []
        prev_id = None
        for p in parsed.parses:
            # Map Montague verbs to strict task frames
            if p.verb in ("create", "modify", "research"):
                task = TransformTask(
                    description=p.instruction,
                    tool_hint="python" if p.verb != "research" else "bash"
                )
            elif p.verb == "read":
                task = QueryTask(description=p.instruction, target_resource=p.target)
            else:
                task = VerifyTask(description=p.instruction, assertion=p.target)

            if prev_id:
                task.preconditions = [prev_id]
            tasks.append(task)
            prev_id = task.id
            
        return TaskDAG(
            user_prompt=user_prompt, 
            intent_observation=parsed.intent_observation, 
            tasks=tasks
        )
        
    except Exception as exc:
        logger.warning("Montague decomposition failed (%s) — falling back.", exc)
        return TaskDAG(
            user_prompt=user_prompt,
            intent_observation="Fallback: single task.",
            tasks=[TransformTask(description=user_prompt[:500], tool_hint="python")]
        )
