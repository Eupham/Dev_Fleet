"""Montague Parser & Fillmore Frame Mapper."""
import uuid
import logging
from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("dev_fleet.task_parser")

class _TaskBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    tool_hint: Literal["bash", "python", ""] = "python"
    preconditions: list[str] = []

class TransformTask(_TaskBase): 
    task_type: Literal["transform"] = "transform"

class QueryTask(_TaskBase): 
    task_type: Literal["query"] = "query"
    target_resource: str = ""

class VerifyTask(_TaskBase): 
    task_type: Literal["verify"] = "verify"
    assertion: str = ""

AtomicTaskNode = Annotated[Union[QueryTask, TransformTask, VerifyTask], Field(discriminator="task_type")]

class TaskDAG(BaseModel):
    user_prompt: str = ""
    intent_observation: str = ""
    tasks: list[AtomicTaskNode]

class MontagueAction(BaseModel):
    verb: Literal["create", "modify", "read", "verify", "research"]
    target: str
    instruction: str

class MontagueDecomposition(BaseModel):
    intent_observation: str = ""
    parses: list[MontagueAction]

def parse_prompt(user_prompt: str) -> TaskDAG:
    from orchestrator.llm_client import chat_completion
    
    try:
        raw_response = chat_completion(
            [
                {"role": "system", "content": "You are a Montague Parser. Extract flat logical actions into JSON."},
                {"role": "user", "content": user_prompt}
            ], 
            schema=MontagueDecomposition
        )
        
        # FIXED: Robust re-instantiation to ensure 'parses' attribute exists
        if isinstance(raw_response, MontagueDecomposition):
            parsed = raw_response
        elif isinstance(raw_response, dict):
            parsed = MontagueDecomposition.model_validate(raw_response)
        else:
            raise ValueError(f"Unexpected response type: {type(raw_response)}")

        tasks = []
        prev_id = None
        for p in parsed.parses:
            # Map Montague actions to Fillmore Frames
            task = TransformTask(description=p.instruction)
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
        fallback_task = TransformTask(description=user_prompt)
        return TaskDAG(
            user_prompt=user_prompt,
            intent_observation="Fallback: single task created.",
            tasks=[fallback_task]
        )
