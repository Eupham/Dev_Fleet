"""Typed task decomposition — Montague Grammar and Fillmore Frames."""
from __future__ import annotations
import uuid
import logging
from typing import Annotated, Literal, Union, List

from pydantic import BaseModel, Field

logger = logging.getLogger("dev_fleet.task_parser")

# --- 1. FILLMORE FRAMES (Execution Schemas - Python Side Only) ---
class _TaskBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    tool_hint: Literal["bash", "python", ""] = "python"
    preconditions: List[str] = []

class TransformTask(_TaskBase): task_type: Literal["transform"] = "transform"
class QueryTask(_TaskBase): task_type: Literal["query"] = "query"
class VerifyTask(_TaskBase): task_type: Literal["verify"] = "verify"

AtomicTaskNode = Annotated[Union[QueryTask, TransformTask, VerifyTask], Field(discriminator="task_type")]

class TaskDAG(BaseModel):
    user_prompt: str = ""
    tasks: List[AtomicTaskNode]

# --- 2. MONTAGUE TRANSLATION LAYER (LLM Side Only) ---
# For llama-server GBNF compatibility, we must keep this as simple as possible.

class MontagueAction(BaseModel):
    """A single logical step."""
    # Using simple strings instead of Literals to reduce GBNF branch complexity
    verb: str = Field(description="Must be exactly: create, modify, read, verify, or research")
    target: str = Field(description="The file path or URL.")
    instruction: str = Field(description="What needs to be done.")

class MontagueDecomposition(BaseModel):
    """
    CRITICAL GBNF FIX: 
    1. No intent_observation field. It creates an early exit point.
    2. 'parses' is the only field, forcing the grammar to open the array immediately.
    """
    parses: List[MontagueAction] = Field(..., description="List of actions to perform.")

# Strict Prompt to assist the GBNF
MONTAGUE_SYSTEM = """You are a logical parser. Translate the request into a JSON list of actions.
Valid verbs: create, modify, read, verify, research.

Output EXACTLY this JSON structure:
{
  "parses": [
    {"verb": "create", "target": "main.py", "instruction": "Write code"}
  ]
}"""

def parse_prompt(user_prompt: str, model: str = "llm", codebase_context: str = "") -> TaskDAG:
    from orchestrator.llm_client import chat_completion
    
    messages = [
        {"role": "system", "content": MONTAGUE_SYSTEM},
        {"role": "user", "content": f"Context:\n{codebase_context}\n\nRequest: {user_prompt}"}
    ]

    try:
        # Calls orchestrator/llm_client.py -> Modal RPC -> inference/server.py -> llama-server
        raw_response = chat_completion(messages, model=model, schema=MontagueDecomposition)
        
        # Safe RPC Unpacking
        if hasattr(raw_response, 'parses'):
            parsed_actions = raw_response.parses
        elif isinstance(raw_response, dict):
            parsed_actions = raw_response.get("parses", [])
        else:
             raise ValueError(f"RPC returned invalid shape: {raw_response}")

        tasks = []
        prev_id = None
        for p in parsed_actions:
            # Map strings back to typed Frames
            v = p.verb.lower()
            if v in ("create", "modify", "research"):
                task = TransformTask(description=p.instruction, tool_hint="bash" if v == "research" else "python")
            elif v == "read":
                task = QueryTask(description=p.instruction, target_resource=p.target)
            else:
                task = VerifyTask(description=p.instruction, assertion=p.target)

            if prev_id: task.preconditions = [prev_id]
            tasks.append(task)
            prev_id = task.id
            
        return TaskDAG(user_prompt=user_prompt, tasks=tasks)
        
    except Exception as exc:
        logger.warning(f"Montague decomposition failed: {exc} — falling back.")
        return TaskDAG(user_prompt=user_prompt, tasks=[TransformTask(description=user_prompt)])
