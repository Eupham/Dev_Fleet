"""Agent Loop — Cyclic state machine for the Dev Fleet orchestrator via LangGraph.

Ties together: Frege parser → Qwen3-Reranker → Graph memory →
LLM generation → Sandbox execution → Episodic graph update.

All inference calls use Modal-native RPC (no HTTP, no timeouts).
"""

from __future__ import annotations

import json
import logging
import operator
import re
from typing import Annotated, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.frege_parser import AtomicTaskNode, TaskDAG, parse_prompt
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.llm_client import generate
from orchestrator.rerank_engine import rerank_candidates
from orchestrator.tool_sandbox import SandboxResult, ModalSandboxTool

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

    dag = parse_prompt(state["user_prompt"])

    for task in dag.tasks:
        memory.add_episodic_node(task.id, task.model_dump())
        for dep_id in task.depends_on:
            memory.add_episodic_edge(dep_id, task.id, {"relation": "depends_on"})

    memory.save()
    return {
        "dag": dag,
        "messages": [f"Decomposed prompt into {len(dag.tasks)} tasks."]
    }


def rerank_and_retrieve_node(state: AgentState) -> dict:
    """Score task edges against semantic, procedural, and episodic graphs via Qwen3-Reranker and link above threshold."""
    logger.info("Executing Rerank_and_Retrieve node...")
    memory = TriGraphMemory.load()
    dag = state["dag"]

    update = {}
    if dag:
        candidates = []
        for n in memory.semantic.nodes:
            candidates.append({"id": n, "graph": "semantic", "description": json.dumps(memory.semantic.nodes[n], default=str)})
        for n in memory.procedural.nodes:
            candidates.append({"id": n, "graph": "procedural", "description": json.dumps(memory.procedural.nodes[n], default=str)})
        for n in memory.episodic.nodes:
            # Avoid self-referencing if nodes share IDs somehow (they usually won't)
            candidates.append({"id": n, "graph": "episodic", "description": json.dumps(memory.episodic.nodes[n], default=str)})

        if candidates:
            for task in dag.tasks:
                edges = rerank_candidates(task.id, task.description, candidates)
                for edge in edges:
                    # Find which graph this candidate came from to tag the edge properly
                    graph_type = next((c["graph"] for c in candidates if c["id"] == edge.candidate_id), "unknown")
                    memory.add_episodic_edge(
                        task.id, edge.candidate_id,
                        {"graph": graph_type, "score": edge.score},
                    )
        update["messages"] = ["Reranked and linked episodic tasks to semantic, procedural, and episodic knowledge."]

    memory.save()
    return update


def execute_node(state: AgentState) -> dict:
    """Generate code, execute in sandbox, record outcome."""
    memory = TriGraphMemory.load()
    dag = state["dag"]
    idx = state["current_task_idx"]

    if not dag or idx >= len(dag.tasks):
        # Done
        return {"final_output": {"nodes": [t.model_dump() for t in (dag.tasks if dag else [])], "graphs": memory.to_dict()}}

    task = dag.tasks[idx]

    # Check dependencies
    deps_ok = all(
        memory.episodic.nodes.get(d, {}).get("status") == "success"
        for d in task.depends_on
    )
    if not deps_ok and task.depends_on:
        logger.info("Skipping %s — unmet dependencies", task.id)
        task.status = "failed"
        memory.episodic.nodes[task.id]["status"] = "failed"
        memory.save()
        return {
            "current_task_idx": idx + 1,
            "current_attempt": 1
        }

    logger.info("Executing Task: %s (attempt %d)", task.id, state["current_attempt"])
    memory.episodic.nodes[task.id]["status"] = "running"
    context = memory.build_context(task.id)

    text = generate(context, task.description)

    # Extract code from markdown blocks if tool_hint is python/bash
    code_to_run = text
    if task.tool_hint in ("python", "bash"):
        pattern = rf"```{task.tool_hint}\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code_to_run = match.group(1)

    update = {}
    if task.tool_hint not in ("python", "bash"):
        memory.episodic.nodes[task.id].update(status="success", response=text[:2000])
        task.status = "success"
        update["sandbox_results"] = [SandboxResult(stdout=text[:1000], stderr="", exit_code=0)]
        # Proceed to next task upon success
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1
    else:
        tool = ModalSandboxTool()
        raw_result = tool.forward(code=code_to_run, language=task.tool_hint)
        result = SandboxResult(stdout=raw_result["stdout"], stderr=raw_result["stderr"], exit_code=raw_result["exit_code"])
        update["sandbox_results"] = [result]
        if result.success:
            memory.episodic.nodes[task.id].update(status="success", output=result.stdout[:2000])
            task.status = "success"
            # Proceed to next task upon success
            update["current_task_idx"] = idx + 1
            update["current_attempt"] = 1
        else:
            fail_id = f"{task.id}_fail_{state['current_attempt']}"
            memory.add_episodic_node(fail_id, {
                "type": "FailedExecution", "stderr": result.stderr[:2000],
                "exit_code": result.exit_code, "attempt": state['current_attempt'],
            })
            memory.add_episodic_edge(task.id, fail_id, {"relation": "failed_execution"})
            task.status = "failed"
            logger.warning("Task %s attempt %d failed (exit %d)", task.id, state['current_attempt'], result.exit_code)
            # DO NOT update indices here on failure, let the router route it to Handle_Failure

    memory.save()
    return update


def handle_failure_node(state: AgentState) -> dict:
    """Handle sandbox failures."""
    logger.info("Executing Handle_Failure node...")
    dag = state["dag"]
    idx = state["current_task_idx"]
    task = dag.tasks[idx]
    memory = TriGraphMemory.load()

    new_attempt = state["current_attempt"] + 1
    update = {"current_attempt": new_attempt}

    if new_attempt > MAX_RETRIES:
        memory.episodic.nodes[task.id]["status"] = "failed"
        task.status = "failed"
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

    # In our model, idx is where the pointer is now.
    # If the execution we just ran was a success, idx incremented. Let's check the PREVIOUS task.
    # However, if it failed, idx did not increment.

    # Let's find the task that was just executed. It will be the one at `idx` if it failed (since idx hasn't advanced).
    # If it succeeded, idx has advanced, so the currently pending task is `idx`, and the one that succeeded is `idx - 1`.

    if idx == 0:
        # Edge case: we couldn't have executed anything yet, or we failed on task 0
        task_just_executed = dag.tasks[0]
    elif idx >= len(dag.tasks):
        # We finished all tasks!
        task_just_executed = dag.tasks[-1]
    else:
        # Check if the current task has a status. If it's pending, we advanced. If it's failed, we didn't advance.
        if dag.tasks[idx].status == "pending":
            task_just_executed = dag.tasks[idx - 1]
        else:
            task_just_executed = dag.tasks[idx]

    if task_just_executed.status == "success":
        if idx >= len(dag.tasks):
            return END
        return "Execute"

    if task_just_executed.status == "failed" and state["current_attempt"] <= MAX_RETRIES:
        # It just failed the execute block, let's route to Handle_Failure
        return "Handle_Failure"

    # If it failed completely and exhausted retries (or skipped due to deps), we move on
    if idx >= len(dag.tasks):
        return END
    return "Execute"


def routing_after_failure(state: AgentState) -> str:
    if state["current_task_idx"] >= len(state["dag"].tasks):
        return END
    return "Execute"


# ---------------------------------------------------------------------------
# Construct the StateGraph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("Decompose", decompose_node)
    workflow.add_node("Rerank_and_Retrieve", rerank_and_retrieve_node)
    workflow.add_node("Execute", execute_node)
    workflow.add_node("Handle_Failure", handle_failure_node)

    workflow.add_edge(START, "Decompose")
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

    update_queue = queue.Queue()

    def run_graph():
        for s in app.stream(initial_state, config=config):
            update_queue.put(("update", s))
        update_queue.put(("done", None))

    t = threading.Thread(target=run_graph)
    t.start()

    while True:
        try:
            msg_type, data = update_queue.get(timeout=30.0)
            if msg_type == "done":
                break

            node_name = list(data.keys())[0]
            state_data = data[node_name]

            # Load memory to get the current graph state
            memory = TriGraphMemory.load()
            yield {
                "step": node_name,
                "state_snapshot": state_data,
                "graphs": memory.to_dict()
            }
        except queue.Empty:
            # Yield a keep-alive message every 30 seconds
            memory = TriGraphMemory.load()
            yield {
                "step": "keep-alive",
                "state_snapshot": {"messages": ["Waiting for tasks to complete..."]},
                "graphs": memory.to_dict()
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
