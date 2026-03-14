"""Agent Loop — Cyclic state machine for the Dev Fleet orchestrator via LangGraph.

Ties together: Frege parser → Qwen3-Reranker → Graph memory →
LLM generation → Sandbox execution → Episodic graph update.

All inference calls use Modal-native RPC (no HTTP, no timeouts).
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import warnings
from typing import Annotated, Any, List, Optional, TypedDict

# Silence LangGraph custom-type serialization warnings (non-fatal, cosmetic noise)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="langgraph.checkpoint.serde.jsonplus",
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.frege_parser import AtomicTaskNode, TaskDAG, parse_prompt
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.llm_client import generate
from orchestrator.rerank_engine import rerank_candidates
from orchestrator.tool_sandbox import SandboxResult, ModalSandboxTool
from orchestrator.supervisor import supervisor_node, conversation_node, direct_execute_node
from orchestrator.codebase_rag import retrieve_codebase_node

logger = logging.getLogger("dev_fleet.agent_loop")

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# TypedDict State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The state passed between nodes in the LangGraph."""
    user_prompt: str
    messages: Annotated[List[str], operator.add]
    dag: Optional[TaskDAG]
    # Semantic supervisor routing fields
    intent: Optional[str]
    codebase_context: str
    # Memory is complex to pass natively via deepcopy in langgraph if it has unpickleable bits
    # We will instantiate or load inside nodes and persist at end, or keep minimal metadata.
    # To keep the state clean, we track which tasks are done:
    current_task_idx: int
    current_attempt: int
    sandbox_results: Annotated[List[SandboxResult], operator.add]
    final_output: Optional[dict]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def decompose_node(state: AgentState) -> dict:
    """Decompose user prompt into Task DAG."""
    logger.info("Executing Decompose node...")
    memory = TriGraphMemory.load()

    dag = parse_prompt(state["user_prompt"], codebase_context=state.get("codebase_context", ""))

    for task in dag.tasks:
        memory.add_episodic_node(task.id, task.model_dump())
        for dep_id in task.depends_on:
            memory.add_episodic_edge(dep_id, task.id, {"relation": "depends_on"})

    memory.save()
    return {
        "dag": dag.model_dump(),
        "messages": [f"Decomposed prompt into {len(dag.tasks)} tasks."]
    }


def rerank_and_retrieve_node(state: AgentState) -> dict:
    """Score task edges against semantic/procedural graphs via a Retrieve-then-Rerank pipeline.

    Stage 1 (Bi-encoder ANN): Uses the PropertyGraphIndex retriever with the fast Qwen3
    embedding model to narrow the full graph vertex set down to a top-k neighbourhood.
    Stage 2 (Cross-encoder): Runs Qwen3-Reranker only over the retrieved neighbourhood,
    reducing complexity from O(V×T) to O(k×T) where k << V.
    """
    logger.info("Executing Rerank_and_Retrieve node...")
    memory = TriGraphMemory.load()
    dag = state["dag"]

    update = {}
    if dag:
        for task in dag.get("tasks", []):
            # Stage 1 — Bi-encoder: retrieve top-15 semantically similar nodes
            retriever = memory.property_graph.as_retriever(similarity_top_k=15)
            retrieved_nodes = retriever.retrieve(task.get("description", ""))

            candidates = []
            for node in retrieved_nodes:
                node_id = node.metadata.get("node_id")
                graph_type = node.metadata.get("graph_type")
                if node_id:
                    candidates.append({
                        "id": node_id,
                        "graph": graph_type or "unknown",
                        "description": node.text,
                    })

            # Stage 2 — Cross-encoder: score only the retrieved neighbourhood
            if candidates:
                edges = rerank_candidates(task.get("id"), task.get("description", ""), candidates)
                for edge in edges:
                    graph_type = next(
                        (c["graph"] for c in candidates if c["id"] == edge.candidate_id),
                        "unknown",
                    )
                    memory.add_episodic_edge(
                        task.get("id"), edge.candidate_id,
                        {"graph": graph_type, "score": edge.score},
                    )

        update["messages"] = ["Reranked and linked episodic tasks to semantic and procedural knowledge."]

    memory.save()
    return update


def execute_node(state: AgentState) -> dict:
    """Generate code, execute in sandbox, record outcome."""
    memory = TriGraphMemory.load()
    dag = state["dag"]
    idx = state["current_task_idx"]

    if not dag or idx >= len(dag.get("tasks", [])):
        # Done
        return {"final_output": {"nodes": dag.get("tasks", []) if dag else [], "graphs": memory.to_dict()}}

    task = dag.get("tasks", [])[idx]

    # Check dependencies
    deps_ok = all(
        memory.episodic.nodes.get(d, {}).get("status") == "success"
        for d in task.get("depends_on", [])
    )
    if not deps_ok and task.get("depends_on", []):
        logger.info("Skipping %s — unmet dependencies", task.get("id"))
        task["status"] = "failed"
        memory.episodic.nodes[task.get("id")]["status"] = "failed"
        memory.save()
        return {
            "current_task_idx": idx + 1,
            "current_attempt": 1
        }

    logger.info("Executing Task: %s (attempt %d)", task.get("id"), state["current_attempt"])
    memory.episodic.nodes[task.get("id")]["status"] = "running"
    graph_context = memory.build_context(task.get("id"))
    # Append sandbox capability hint so the LLM knows it can read files via shell
    context = (
        graph_context
        + "\n\n[EXECUTION ENVIRONMENT] You are running inside an isolated Modal Sandbox. "
        "You do NOT have the file contents yet. Use bash (`cat`, `grep`, `find`) or Python "
        "(`open()`, `pathlib`) to read files and explore the repository before making changes."
    )

    tool_hint = task.get("tool_hint", "")
    temp = 0.3 if tool_hint in ("python", "bash") else 0.7
    text = generate(context, task.get("description", ""), temperature=temp)

    # Extract code from markdown blocks if tool_hint is python/bash
    code_to_run = text
    if tool_hint in ("python", "bash"):
        pattern = rf"```{tool_hint}\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code_to_run = match.group(1)

    update = {}
    if tool_hint not in ("python", "bash"):
        memory.episodic.nodes[task.get("id")].update(status="success", response=text[:2000])
        task["status"] = "success"
        update["sandbox_results"] = [{"stdout": text[:1000], "stderr": "", "exit_code": 0}]
        # Proceed to next task upon success
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1
    else:
        tool = ModalSandboxTool()
        raw_result = tool.forward(code=code_to_run, language=tool_hint)
        result = SandboxResult(stdout=raw_result["stdout"], stderr=raw_result["stderr"], exit_code=raw_result["exit_code"])
        update["sandbox_results"] = [{"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}]
        if result.success:
            memory.episodic.nodes[task.get("id")].update(status="success", output=result.stdout[:2000])
            task["status"] = "success"
            # Proceed to next task upon success
            update["current_task_idx"] = idx + 1
            update["current_attempt"] = 1
        else:
            fail_id = f"{task.get('id')}_fail_{state['current_attempt']}"
            memory.add_episodic_node(fail_id, {
                "type": "FailedExecution", "stderr": result.stderr[:2000],
                "exit_code": result.exit_code, "attempt": state['current_attempt'],
            })
            memory.add_episodic_edge(task.get("id"), fail_id, {"relation": "failed_execution"})
            task["status"] = "failed"
            logger.warning("Task %s attempt %d failed (exit %d)", task.get("id"), state['current_attempt'], result.exit_code)
            # DO NOT update indices here on failure, let the router route it to Handle_Failure

    memory.save()
    return update


def handle_failure_node(state: AgentState) -> dict:
    """Handle sandbox failures."""
    logger.info("Executing Handle_Failure node...")
    dag = state["dag"]
    idx = state["current_task_idx"]
    task = dag.get("tasks", [])[idx]
    memory = TriGraphMemory.load()

    new_attempt = state["current_attempt"] + 1
    update = {"current_attempt": new_attempt}

    if new_attempt > MAX_RETRIES:
        memory.episodic.nodes[task.get("id")]["status"] = "failed"
        task["status"] = "failed"
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1

    memory.save()
    return update


def routing_after_execute(state: AgentState) -> str:
    """Determine the next step based on execution outcome."""
    dag = state["dag"]
    idx = state["current_task_idx"]

    # We must look at the previous task to see if it failed, because if it succeeded, idx has already incremented
    if not dag:
        return END

    tasks = dag.get("tasks", [])

    if idx == 0:
        task_just_executed = tasks[0]
    elif idx >= len(tasks):
        task_just_executed = tasks[-1]
    else:
        if tasks[idx].get("status") == "pending":
            task_just_executed = tasks[idx - 1]
        else:
            task_just_executed = tasks[idx]

    if task_just_executed.get("status") == "success":
        if idx >= len(tasks):
            return END
        return "Execute"

    if task_just_executed.get("status") == "failed" and state["current_attempt"] <= MAX_RETRIES:
        return "Handle_Failure"

    if idx >= len(tasks):
        return END
    return "Execute"


def routing_after_failure(state: AgentState) -> str:
    if state["current_task_idx"] >= len(state["dag"].get("tasks", [])):
        return END
    return "Execute"


# ---------------------------------------------------------------------------
# Construct the StateGraph
# ---------------------------------------------------------------------------

def route_from_supervisor(state: AgentState) -> str:
    """Route based on the intent field set by supervisor_node."""
    intent = state.get("intent", "DECOMPOSE")
    if intent == "CONVERSATION":
        return "Conversation"
    if intent == "DIRECT_EXECUTE":
        return "Direct_Execute"
    return "Retrieve_Codebase"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # --- Semantic supervisor layer ---
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Conversation", conversation_node)
    workflow.add_node("Direct_Execute", direct_execute_node)
    workflow.add_node("Retrieve_Codebase", retrieve_codebase_node)

    # --- Existing execution pipeline ---
    workflow.add_node("Decompose", decompose_node)
    workflow.add_node("Rerank_and_Retrieve", rerank_and_retrieve_node)
    workflow.add_node("Execute", execute_node)
    workflow.add_node("Handle_Failure", handle_failure_node)

    # Entry point
    workflow.add_edge(START, "Supervisor")

    # Supervisor → branch on intent
    workflow.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {
            "Retrieve_Codebase": "Retrieve_Codebase",
            "Conversation": "Conversation",
            "Direct_Execute": "Direct_Execute",
        },
    )

    # Conversation → END (simple reply, no execution)
    workflow.add_edge("Conversation", END)

    # Direct_Execute → Execute (single-task DAG, skip decomposition)
    workflow.add_edge("Direct_Execute", "Execute")

    # RAG → Decompose → existing pipeline
    workflow.add_edge("Retrieve_Codebase", "Decompose")
    workflow.add_edge("Decompose", "Rerank_and_Retrieve")
    workflow.add_edge("Rerank_and_Retrieve", "Execute")

    workflow.add_conditional_edges("Execute", routing_after_execute)
    workflow.add_conditional_edges("Handle_Failure", routing_after_failure)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public Entrypoint
# ---------------------------------------------------------------------------

def agent_loop_stream(user_prompt: str):
    """Run the full agent loop via LangGraph and stream the steps out."""

    initial_state: AgentState = {
        "user_prompt": user_prompt,
        "messages": [],
        "dag": None,
        "intent": None,
        "codebase_context": "",
        "current_task_idx": 0,
        "current_attempt": 1,
        "sandbox_results": [],
        "final_output": None
    }

    app = build_graph()

    # We use a static thread_id for this single synchronous loop run
    config = {"configurable": {"thread_id": "1"}}

    # Stream over nodes for real-time visibility
    import threading
    import queue
    import time

    update_queue: queue.Queue = queue.Queue()

    def run_graph():
        try:
            for s in app.stream(initial_state, config=config):
                node_name = list(s.keys())[0]
                s_update = s[node_name]
                # Fetch the accumulated checkpoint state
                full_state = app.get_state(config).values
                # Merge s_update into safe_state — scalar fields from s_update take
                # precedence to avoid stale-checkpoint timing issues (e.g. intent=null).
                safe_state = {
                    "user_prompt": full_state.get("user_prompt", initial_state["user_prompt"]),
                    "messages": full_state.get("messages", []),
                    "dag": s_update["dag"] if "dag" in s_update else full_state.get("dag"),
                    "intent": s_update["intent"] if "intent" in s_update else full_state.get("intent"),
                    "codebase_context": s_update["codebase_context"] if "codebase_context" in s_update else full_state.get("codebase_context", ""),
                    "current_task_idx": s_update.get("current_task_idx", full_state.get("current_task_idx", 0)),
                    "current_attempt": s_update.get("current_attempt", full_state.get("current_attempt", 1)),
                    "sandbox_results": full_state.get("sandbox_results", []),
                    "final_output": s_update["final_output"] if "final_output" in s_update else full_state.get("final_output"),
                }
                update_queue.put(("update", node_name, safe_state, s_update))
            update_queue.put(("done", None, None, None))
        except Exception as exc:
            update_queue.put(("error", exc, None, None))

    t = threading.Thread(target=run_graph, daemon=True)
    t.start()

    while True:
        try:
            msg = update_queue.get(timeout=30.0)
            msg_type = msg[0]

            if msg_type == "done":
                break
            if msg_type == "error":
                raise msg[1]

            _, node_name, state_data, node_update = msg

            # Load memory to get the current graph state
            memory = TriGraphMemory.load()
            yield {
                "step": node_name,
                "state_snapshot": state_data,
                "graphs": memory.to_dict(),
                "node_update": node_update,
            }
        except queue.Empty:
            # Yield a keep-alive message every 30 seconds to keep Modal stream open
            memory = TriGraphMemory.load()
            yield {
                "step": "keep-alive",
                "state_snapshot": {"messages": ["Waiting for tasks to complete..."]},
                "graphs": memory.to_dict(),
                "node_update": {},
            }



def agent_loop(user_prompt: str) -> dict[str, Any]:
    """Run the full agent loop synchronously and return the final graph state."""
    final_output = None
    # Just run through the stream until the end
    for update in agent_loop_stream(user_prompt):
        if update["state_snapshot"].get("final_output"):
            final_output = update["state_snapshot"]["final_output"]

    if final_output:
        return final_output

    memory = TriGraphMemory.load()
    return {
        "nodes": [],
        "graphs": memory.to_dict()
    }
