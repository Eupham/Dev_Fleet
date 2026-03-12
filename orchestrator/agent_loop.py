"""Agent Loop — Cyclic state machine for the Dev Fleet orchestrator.

Ties together: Frege parser → Qwen3-Reranker → Graph memory →
LLM generation → Sandbox execution → Episodic graph update.

All inference calls use Modal-native RPC (no HTTP, no timeouts).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from orchestrator.frege_parser import AtomicTaskNode, TaskDAG, parse_prompt
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.llm_client import generate
from orchestrator.rerank_engine import rerank_candidates
from orchestrator.tool_sandbox import SandboxResult, execute_in_sandbox

logger = logging.getLogger("devfleet.agent_loop")

MAX_RETRIES = 2


def _build_dag(user_prompt: str) -> TaskDAG:
    """Decompose *user_prompt* into a Task DAG."""
    return parse_prompt(user_prompt)


def _rerank_and_link(memory: TriGraphMemory, dag: TaskDAG) -> None:
    """Score task↔semantic edges via Qwen3-Reranker and link above threshold."""
    candidates = [
        {"id": n, "description": json.dumps(memory.semantic.nodes[n], default=str)}
        for n in memory.semantic.nodes
    ]
    if not candidates:
        return
    for task in dag.tasks:
        for edge in rerank_candidates(task.id, task.description, candidates):
            memory.add_episodic_edge(
                task.id, edge.candidate_id,
                {"graph": "semantic", "score": edge.score},
            )


def _execute_task(task: AtomicTaskNode, memory: TriGraphMemory) -> None:
    """Generate code, execute in sandbox, record outcome."""
    memory.episodic.nodes[task.id]["status"] = "running"
    context = memory.build_context(task.id)

    for attempt in range(1, MAX_RETRIES + 1):
        text = generate(context, task.description)
        if task.tool_hint not in ("python", "bash"):
            memory.episodic.nodes[task.id].update(status="success", response=text[:2000])
            task.status = "success"
            return
        result: SandboxResult = execute_in_sandbox(text, language=task.tool_hint)
        if result.success:
            memory.episodic.nodes[task.id].update(status="success", output=result.stdout[:2000])
            task.status = "success"
            return
        fail_id = f"{task.id}_fail_{attempt}"
        memory.add_episodic_node(fail_id, {
            "type": "FailedExecution", "stderr": result.stderr[:2000],
            "exit_code": result.exit_code, "attempt": attempt,
        })
        memory.add_episodic_edge(task.id, fail_id, {"relation": "failed_execution"})
        logger.warning("Task %s attempt %d failed (exit %d)", task.id, attempt, result.exit_code)
    memory.episodic.nodes[task.id]["status"] = "failed"
    task.status = "failed"


def agent_loop(user_prompt: str) -> dict[str, Any]:
    """Run the full agent loop and return the final graph state."""
    memory = TriGraphMemory.load()

    dag = _build_dag(user_prompt)
    for task in dag.tasks:
        memory.add_episodic_node(task.id, task.model_dump())
        for dep_id in task.depends_on:
            memory.add_episodic_edge(dep_id, task.id, {"relation": "depends_on"})

    _rerank_and_link(memory, dag)

    for task in dag.tasks:
        deps_ok = all(
            memory.episodic.nodes.get(d, {}).get("status") == "success"
            for d in task.depends_on
        )
        if not deps_ok and task.depends_on:
            logger.info("Skipping %s — unmet dependencies", task.id)
            continue
        _execute_task(task, memory)

    memory.save()
    return {"nodes": [t.model_dump() for t in dag.tasks], "graphs": memory.to_dict()}
