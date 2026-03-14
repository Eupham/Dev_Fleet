"""Supervisor — Intent classification and routing node.

Separates the concern of *what the user wants* from *how to do it*.
Three intents are recognised:

  CONVERSATION    — greetings, questions, chat.  Handled by conversation_node.
  DECOMPOSE       — multi-step coding tasks.  Routed to Retrieve_Codebase → Decompose.
  DIRECT_EXECUTE  — simple, single-step commands.  Bypasses Frege decomposition.
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
    intent: Literal["CONVERSATION", "DECOMPOSE", "DIRECT_EXECUTE"] = Field(
        description=(
            "Classify the prompt. "
            "CONVERSATION for greetings/chat. "
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

    prompt_lower = state["user_prompt"].lower()

    # Hard-coded override: research or build prompts always decompose.
    # This prevents small models from misrouting multi-step tasks.
    _decompose_signals = (
        "research", "build", "create", "write", "implement", "develop",
        "design", "make", "code", "program", "find", "search", "look up",
        "analyze", "test", "fix", "debug", "deploy", "run", "execute",
        "domain", "industry", "application", "solve", "generate",
    )
    _conversation_only = (
        "hello", "hi ", "hey ", "thanks", "thank you", "what is your name",
        "how are you", "who are you",
    )

    is_clearly_conversational = any(prompt_lower.startswith(s) or s == prompt_lower for s in _conversation_only)
    has_action_word = any(w in prompt_lower for w in _decompose_signals)

    if has_action_word and not is_clearly_conversational:
        intent = "DECOMPOSE"
        logger.info("Supervisor override: action words detected, forcing DECOMPOSE.")
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier. "
                    "Respond with exactly one word — no explanation, no punctuation.\n"
                    "CONVERSATION: greetings, smalltalk, asking who you are.\n"
                    "DIRECT_EXECUTE: a single simple command (e.g. 'print hello world').\n"
                    "DECOMPOSE: everything else — research, build, multi-step tasks."
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
        "messages": [f"[SUPERVISOR] Intent: {intent}"],
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
        "messages": [reply_text],
    }


def direct_execute_node(state: "AgentState") -> dict:
    """Bypass Frege decomposition and create a single-task DAG directly."""
    import logging
    from orchestrator.frege_parser import AtomicTaskNode, TaskDAG
    from orchestrator.graph_memory import TriGraphMemory

    logger = logging.getLogger("dev_fleet.supervisor")
    logger.info("Direct_Execute node bypassing Frege decomposition.")

    single_task = AtomicTaskNode(
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
        "messages": ["Bypassed decomposition for direct execution."],
    }
