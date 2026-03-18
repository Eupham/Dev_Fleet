"""Supervisor — Intent classification and routing node.

Separates the concern of *what the user wants* from *how to do it*.
Four intents are recognised:

  CONVERSATION    — greetings, questions, chat.  Handled by conversation_node.
  RESEARCH        — web research requests.  Routed to Research → Decompose.
  DECOMPOSE       — multi-step coding tasks.  Routed to Retrieve_Codebase → Decompose.
  DIRECT_EXECUTE  — simple, single-step commands.  Bypasses LLM decomposition.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from orchestrator.agent_loop import AgentState


# ---------------------------------------------------------------------------
# Intent schema
# ---------------------------------------------------------------------------


class IntentClassification(BaseModel):
    intent: Literal["CONVERSATION", "RESEARCH", "DECOMPOSE", "DIRECT_EXECUTE"] = Field(
        description=(
            "Classify the prompt. "
            "CONVERSATION for greetings/chat. "
            "RESEARCH for web research, online lookups, finding information. "
            "DECOMPOSE for complex coding tasks requiring multiple steps. "
            "DIRECT_EXECUTE for simple, single-step commands."
        )
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def supervisor_node(state: "AgentState") -> dict:
    import logging
    from orchestrator.llm_client import chat_completion

    logger = logging.getLogger("dev_fleet.supervisor")
    logger.info("Supervisor classifying intent...")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. "
                "Respond with exactly one word — no explanation, no punctuation.\n"
                "CONVERSATION: greetings, smalltalk, asking who you are.\n"
                "RESEARCH: requests to search online, do web research, look up information.\n"
                "DIRECT_EXECUTE: a single simple command (e.g. 'print hello world').\n"
                "DECOMPOSE: everything else — build, create, implement, multi-step tasks."
            ),
        },
        {"role": "user", "content": state["user_prompt"]},
    ]
    try:
        classification: IntentClassification = chat_completion(
            messages,
            temperature=0.0,
            max_tokens=16,
            schema=IntentClassification,
        )
        intent = classification.intent
    except Exception as exc:
        logger.warning("Intent classification failed (%s). Defaulting to DECOMPOSE.", exc)
        intent = "DECOMPOSE"

    logger.info("Supervisor classified intent as %s", intent)
    return {
        "intent": intent,
        "messages": [{"role": "assistant", "content": f"[SUPERVISOR] Intent: {intent}"}],
    }


def conversation_node(state: "AgentState") -> dict:
    """Handle conversational prompts with a simple LLM reply."""
    import logging
    from orchestrator.llm_client import chat_completion

    logger = logging.getLogger("dev_fleet.supervisor")
    logger.info("Conversation node handling chat prompt.")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Reply concisely."},
        {"role": "user", "content": state["user_prompt"]},
    ]

    try:
        reply_text: str = chat_completion(messages, temperature=0.7, max_tokens=512)
    except Exception as exc:
        logger.warning("Conversation generation failed (%s).", exc)
        reply_text = "I'm here to help. Could you tell me more about what you need?"

    return {
        "final_output": {"response": reply_text},
        "messages": [{"role": "assistant", "content": reply_text}],
    }


def direct_execute_node(state: "AgentState") -> dict:
    """Bypass LLM decomposition and create a single-task DAG directly."""
    import logging
    from orchestrator.task_parser import AtomicTaskNode, TaskDAG, TransformTask
    from orchestrator.graph_memory import TriGraphMemory

    logger = logging.getLogger("dev_fleet.supervisor")
    logger.info("Direct_Execute node bypassing LLM decomposition.")

    single_task = TransformTask(
        id=uuid.uuid4().hex[:12],
        description=state["user_prompt"][:500],
        tool_hint="python",
    )
    dag = TaskDAG(
        user_prompt=state["user_prompt"],
        intent_observation="Direct single-step execution — decomposition bypassed.",
        tasks=[single_task],
    )

    # Register the task in episodic memory so execute_node can find it by ID.
    memory = TriGraphMemory.load()
    memory.add_episodic_node(single_task.id, single_task.model_dump())
    memory.save()

    return {
        "dag": dag.model_dump(),
        "messages": [{"role": "assistant", "content": "Bypassed decomposition for direct execution."}],
    }
