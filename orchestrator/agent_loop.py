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
import uuid
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
    next_route: Optional[str]


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

    Wrapped in a top-level try/except: if the embedder cold-starts, times out,
    or is otherwise unavailable, this node degrades gracefully and the graph
    continues to Execute rather than silently ending the LangGraph stream.
    """
    logger.info("Executing Rerank_and_Retrieve node...")
    try:
        memory = TriGraphMemory.load()
        dag = state["dag"]

        # Skip retrieval if the semantic/procedural graphs are empty.
        # Calling the embedder against an empty index wastes an RPC round-trip
        # and can cause the graph to hang on cold-starts.
        pg_has_nodes = (
            memory.semantic.number_of_nodes() > 0
            or memory.procedural.number_of_nodes() > 0
        )

        if dag and pg_has_nodes:
            for task in dag.get("tasks", []):
                try:
                    # Stage 1 — Bi-encoder: retrieve top-15 semantically similar nodes
                    retriever = memory.property_graph.as_retriever(similarity_top_k=15)
                    retrieved_nodes = retriever.retrieve(task.get("description", ""))
                except Exception as retrieval_exc:
                    logger.warning("Retrieval failed for task %s (%s) — skipping.", task.get("id"), retrieval_exc)
                    continue

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
                    try:
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
                    except Exception as rerank_exc:
                        logger.warning("Reranking failed for task %s (%s) — skipping.", task.get("id"), rerank_exc)

            memory.save()
            return {"messages": ["Reranked and linked episodic tasks to semantic and procedural knowledge."]}

        # Empty graphs — skip linking, proceed immediately to Execute
        return {"messages": ["Knowledge graphs empty — skipping linking, proceeding to execution."]}

    except Exception as exc:
        logger.warning("Rerank_and_Retrieve failed (%s) — skipping knowledge linking.", exc)
        return {"messages": [f"Knowledge linking skipped ({type(exc).__name__}) — proceeding to execution."]}


def _append_workspace_entry(
    current_ctx: str,
    task_desc: str,
    tool_hint: str,
    stdout: str,
    stderr: str,
    exit_code: int,
) -> str:
    """Append a one-line workspace summary entry to the running codebase_context."""
    status = "✓" if exit_code == 0 else "✗"
    lang = tool_hint if tool_hint else "text"
    output_snippet = (stdout.strip()[:200] + "...") if len(stdout.strip()) > 200 else stdout.strip()
    if not output_snippet and stderr.strip():
        output_snippet = f"[stderr] {stderr.strip()[:300]}"
    entry = f"{status} [{lang}] {task_desc[:80]}"
    if output_snippet:
        entry += f"\n    → {output_snippet}"
    lines = [l for l in current_ctx.split("\n") if l.strip()]
    lines.append(entry)
    return "\n".join(lines)


def _with_task_status(dag_dict: dict, idx: int, status: str) -> dict:
    """Return a new dag dict with tasks[idx].status updated. Never mutates in place."""
    tasks = [t.copy() for t in dag_dict.get("tasks", [])]
    tasks[idx] = {**tasks[idx], "status": status}
    return {**dag_dict, "tasks": tasks}


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
        memory.episodic.nodes[task.get("id")]["status"] = "failed"
        memory.save()
        return {
            "dag": _with_task_status(dag, idx, "failed"),
            "current_task_idx": idx + 1,
            "current_attempt": 1,
            "next_route": "Execute" if idx + 1 < len(dag.get("tasks", [])) else "__end__",
        }

    logger.info("Executing Task: %s (attempt %d)", task.get("id"), state["current_attempt"])
    memory.episodic.nodes[task.get("id")]["status"] = "running"

    # build_context calls the embedder — wrap so a cold-start failure doesn't crash the node
    try:
        graph_context = memory.build_context(task.get("id"))
    except Exception as ctx_exc:
        logger.warning("build_context failed for %s (%s) — using empty context.", task.get("id"), ctx_exc)
        graph_context = "(no context)"

    tool_hint = task.get("tool_hint", "")
    context = (
        graph_context
        + "\n\n[EXECUTION ENVIRONMENT] You are running inside an isolated Modal Sandbox. "
        "Files persist to /workspace between tasks. "
        "Write all output files to /workspace/. "
        "Use bash (`cat`, `grep`, `find`, `ls /workspace`) or Python "
        "(`open()`, `pathlib`) to read and explore files already in /workspace."
    )

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
        update["dag"] = _with_task_status(dag, idx, "success")
        update["sandbox_results"] = [{"stdout": text[:1000], "stderr": "", "exit_code": 0, "code": "", "task_description": task.get("description", ""), "tool_hint": tool_hint}]
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1
        update["next_route"] = "Execute" if idx + 1 < len(dag.get("tasks", [])) else "__end__"
        # Update workspace context with task result
        update["codebase_context"] = _append_workspace_entry(
            state.get("codebase_context", ""), task.get("description", ""), tool_hint, text[:300], "", 0
        )
    else:
        tool = ModalSandboxTool()
        raw_result = tool.forward(code=code_to_run, language=tool_hint)
        result = SandboxResult(stdout=raw_result["stdout"], stderr=raw_result["stderr"], exit_code=raw_result["exit_code"])
        update["sandbox_results"] = [{"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code, "code": code_to_run[:2000], "task_description": task.get("description", ""), "tool_hint": tool_hint}]
        # Update workspace context after every sandbox execution
        update["codebase_context"] = _append_workspace_entry(
            state.get("codebase_context", ""),
            task.get("description", ""),
            tool_hint,
            result.stdout,
            result.stderr,
            result.exit_code,
        )
        if result.success:
            memory.episodic.nodes[task.get("id")].update(status="success", output=result.stdout[:2000])
            update["dag"] = _with_task_status(dag, idx, "success")
            update["current_task_idx"] = idx + 1
            update["current_attempt"] = 1
            update["next_route"] = "Execute" if idx + 1 < len(dag.get("tasks", [])) else "__end__"
        else:
            fail_id = f"{task.get('id')}_fail_{state['current_attempt']}"
            memory.add_episodic_node(fail_id, {
                "type": "FailedExecution", "stderr": result.stderr[:2000],
                "exit_code": result.exit_code, "attempt": state['current_attempt'],
            })
            memory.add_episodic_edge(task.get("id"), fail_id, {"relation": "failed_execution"})
            update["dag"] = _with_task_status(dag, idx, "failed")
            update["next_route"] = "Handle_Failure"
            logger.warning("Task %s attempt %d failed (exit %d)", task.get("id"), state['current_attempt'], result.exit_code)

    memory.save()
    return update


def handle_failure_node(state: AgentState) -> dict:
    """Handle sandbox failures — extract error context for retry visibility."""
    logger.info("Executing Handle_Failure node...")
    dag = state["dag"]
    idx = state["current_task_idx"]
    task = dag.get("tasks", [])[idx]
    memory = TriGraphMemory.load()

    # Extract the last sandbox error for UI visibility
    last_stderr = ""
    last_exit_code = 1
    sandbox_results = state.get("sandbox_results", [])
    if sandbox_results:
        last = sandbox_results[-1]
        if isinstance(last, dict):
            last_stderr = last.get("stderr", "")
            last_exit_code = last.get("exit_code", 1)
        else:
            last_stderr = getattr(last, "stderr", "")
            last_exit_code = getattr(last, "exit_code", 1)

    new_attempt = state["current_attempt"] + 1
    update: dict = {
        "current_attempt": new_attempt,
        "messages": [
            f"Retrying task (attempt {new_attempt} of {MAX_RETRIES})...\n"
            f"Stderr: {last_stderr[:500] if last_stderr else '(empty)'}\n"
            f"Exit code: {last_exit_code}"
        ],
    }

    if new_attempt > MAX_RETRIES:
        memory.episodic.nodes[task.get("id")]["status"] = "failed"
        dag = state["dag"]
        tasks = [t.copy() for t in dag.get("tasks", [])]
        tasks[idx] = {**tasks[idx], "status": "failed"}
        update["dag"] = {**dag, "tasks": tasks}
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1
        update["next_route"] = "Execute" if idx + 1 < len(tasks) else "__end__"
        logger.warning("Task %s exhausted retries — marking failed. Last error: %s", task.get("id"), last_stderr[:200])
    else:
        update["next_route"] = "Execute"

    memory.save()
    return update


def routing_after_execute(state: AgentState) -> str:
    """Determine the next step based on execution outcome."""
    route = state.get("next_route") or "__end__"
    dag = state.get("dag")
    idx = state.get("current_task_idx", 0)
    if route == "__end__" or not dag or idx >= len(dag.get("tasks", [])):
        return END
    return route


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


_compiled_graph: Any = None


def get_compiled_graph():
    """Return the module-level compiled graph, building it once on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public Entrypoint
# ---------------------------------------------------------------------------

async def agent_loop_stream(user_prompt: str):
    """Run the full agent loop via LangGraph and stream the steps out asynchronously."""

    initial_state: AgentState = {
        "user_prompt": user_prompt,
        "messages": [],
        "dag": None,
        "intent": None,
        "codebase_context": "",
        "current_task_idx": 0,
        "current_attempt": 1,
        "sandbox_results": [],
        "final_output": None,
        "next_route": None,
    }

    app = get_compiled_graph()

    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}

    try:
        async for s in app.astream(initial_state, config=config):
            node_name = list(s.keys())[0]
            s_update = s[node_name]
            full_state = await app.aget_state(config)
            full_state_values = full_state.values

            safe_state = {
                "user_prompt": full_state_values.get("user_prompt", initial_state["user_prompt"]),
                "messages": full_state_values.get("messages", []),
                "dag": s_update["dag"] if "dag" in s_update else full_state_values.get("dag"),
                "intent": s_update["intent"] if "intent" in s_update else full_state_values.get("intent"),
                "codebase_context": s_update["codebase_context"] if "codebase_context" in s_update else full_state_values.get("codebase_context", ""),
                "current_task_idx": s_update.get("current_task_idx", full_state_values.get("current_task_idx", 0)),
                "current_attempt": s_update.get("current_attempt", full_state_values.get("current_attempt", 1)),
                "sandbox_results": full_state_values.get("sandbox_results", []),
                "final_output": s_update["final_output"] if "final_output" in s_update else full_state_values.get("final_output"),
                "next_route": s_update.get("next_route", full_state_values.get("next_route")),
            }

            TriGraphMemory.configure()
            memory = TriGraphMemory.load()

            yield {
                "step": node_name,
                "state_snapshot": safe_state,
                "graphs": memory.to_dict(),
                "node_update": s_update,
            }

    except GeneratorExit:
        logger.info("agent_loop_stream cancelled by caller.")
        pass



def agent_loop(user_prompt: str) -> dict[str, Any]:
    """Run the full agent loop synchronously and return the final graph state."""
    import asyncio

    async def run_sync():
        final_output = None
        async for update in agent_loop_stream(user_prompt):
            if update["state_snapshot"].get("final_output"):
                final_output = update["state_snapshot"]["final_output"]
        return final_output

    final_output = asyncio.run(run_sync())

    if final_output:
        return final_output

    TriGraphMemory.configure()
    memory = TriGraphMemory.load()
    return {
        "nodes": [],
        "graphs": memory.to_dict()
    }
