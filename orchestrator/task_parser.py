"""Typed task decomposition — Montague Grammar and Fillmore Frames."""
from __future__ import annotations
import uuid
import logging
from typing import Annotated, Literal, Union, List
from pydantic import BaseModel, Field

logger = logging.getLogger("dev_fleet.task_parser")

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

class MontagueAction(BaseModel):
    verb: str = Field(description="Must be exactly: create, modify, read, verify, or research")
    target: str = Field(description="The file path or URL.")
    instruction: str = Field(description="What needs to be done.")

class MontagueDecomposition(BaseModel):
    parses: List[MontagueAction] = Field(..., description="List of actions to perform.")

# <--- CRITICAL FIX: ONE-SHOT EXAMPLE ADDED --->
MONTAGUE_SYSTEM = """You are a logical parser. Translate the request into a JSON list of actions.
Valid verbs: create, modify, read, verify, research.

Example Input: "Do online research about quantum computing and write a script."
Example Output:
{
  "parses": [
    {"verb": "research", "target": "quantum computing", "instruction": "Research quantum computing principles"},
    {"verb": "create", "target": "quantum_script.py", "instruction": "Write a python script based on research"}
  ]
}

Respond ONLY with valid JSON matching the exact structure above."""

def parse_prompt(user_prompt: str, model: str = "llm", codebase_context: str = "") -> TaskDAG:
    from orchestrator.llm_client import chat_completion

    messages = [
        {"role": "system", "content": MONTAGUE_SYSTEM},
        {"role": "user", "content": f"Context:\n{codebase_context}\n\nRequest: {user_prompt}"}
    ]

    try:
        raw_response = chat_completion(messages, model=model, schema=MontagueDecomposition)

        # HONEST ASSESSMENT: Handle None return (validation failed)
        if raw_response is None:
            raise ValueError("Schema validation returned None — LLM output did not match expected schema")

        # Extract parsed actions from the response
        if hasattr(raw_response, 'parses'):
            parsed_actions = raw_response.parses
        elif isinstance(raw_response, dict):
            parsed_actions = raw_response.get("parses", [])
        else:
            raise ValueError(f"RPC returned unexpected type: {type(raw_response)}")

        # HONEST ASSESSMENT: Empty parses list means decomposition failed
        if not parsed_actions:
            raise ValueError("LLM returned empty parses list")

        tasks = []
        prev_id = None
        for p in parsed_actions:
            v = p.verb.lower() if hasattr(p, 'verb') else p.get('verb', '').lower()
            instruction = p.instruction if hasattr(p, 'instruction') else p.get('instruction', user_prompt)
            target = p.target if hasattr(p, 'target') else p.get('target', '')

            if v in ("create", "modify", "research"):
                task = TransformTask(description=instruction, tool_hint="bash" if v == "research" else "python")
            elif v == "read":
                task = QueryTask(description=instruction)
            else:
                task = VerifyTask(description=instruction)

            if prev_id:
                task.preconditions = [prev_id]
            tasks.append(task)
            prev_id = task.id

        # Final sanity check
        if not tasks:
            raise ValueError("No tasks were created from parsed actions")

        return TaskDAG(user_prompt=user_prompt, tasks=tasks)

    except Exception as exc:
        logger.warning(f"Montague decomposition failed: {exc} — using single-task fallback")
        # HONEST FALLBACK: Single transform task that captures the full user intent
        return TaskDAG(
            user_prompt=user_prompt,
            tasks=[TransformTask(description=user_prompt, tool_hint="python")]
        )
