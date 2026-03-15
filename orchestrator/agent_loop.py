"""Agent Loop — Cyclic state machine for the Dev Fleet orchestrator via LangGraph.

Ties together: Supervisor → (Research?) → Retrieve_Codebase → Decompose →
Rerank_and_Retrieve → Execute → (Validate?) → END.

All inference calls use Modal-native RPC (no HTTP, no timeouts).
Difficulty is derived from reranker scores + graph topology (no LLM call).
Composition is observed by diffing the sandbox filesystem (not declared).
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import uuid
import warnings
from typing import Annotated, Any, AsyncIterator, List, Optional, TypedDict

# Silence LangGraph custom-type serialization warnings (non-fatal, cosmetic noise)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="langgraph.checkpoint.serde.jsonplus",
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.task_parser import (
    AtomicTaskNode, TaskDAG, parse_prompt,
    TransformTask, QueryTask, VerifyTask, ComposeTask,
)
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.llm_client import generate, chat_completion
from orchestrator.rerank_engine import rerank_candidates, ScoredEdge
from orchestrator.tool_sandbox import SandboxResult, ModalSandboxTool
from orchestrator.supervisor import supervisor_node, conversation_node, direct_execute_node
from orchestrator.codebase_rag import retrieve_codebase_node

logger = logging.getLogger("dev_fleet.agent_loop")

MAX_RETRIES = 2     # per-task retry budget
MAX_ITERATIONS = 5  # outer validation/iteration budget


# ---------------------------------------------------------------------------
# TypedDict State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The state passed between nodes in the LangGraph."""
    user_prompt: str
    messages: Annotated[List[str], operator.add]
    dag: Optional[dict]
    intent: Optional[str]
    codebase_context: str
    current_task_idx: int
    current_attempt: int
    sandbox_results: Annotated[List[dict], operator.add]
    final_output: Optional[dict]
    next_route: Optional[str]
    # Composition ledger data (JSON-serializable)
    composition_deltas: dict
    # Per-task difficulty scores
    task_difficulties: dict
    # Outer iteration counter for validation loop
    iteration_count: int
    # DRS serialized — empty dict = empty scope
    discourse_state: dict
    # DRS for active retry scope — empty dict when none
    retry_discourse_state: dict
    # for execution phase determination
    tasks_completed_count: int


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def decompose_node(state: AgentState) -> dict:
    """Decompose user prompt into Task DAG."""
    logger.info("Executing Decompose node...")
    memory = TriGraphMemory.load()

    is_research = state.get("intent") == "RESEARCH"
    dag = parse_prompt(
        state["user_prompt"],
        codebase_context=state.get("codebase_context", ""),
        is_research=is_research,
    )

    for task in dag.tasks:
        memory.add_episodic_node(task.id, task.model_dump())

    memory.save()
    return {
        "dag": dag.model_dump(),
        "messages": [f"Decomposed prompt into {len(dag.tasks)} tasks."]
    }


def rerank_and_retrieve_node(state: AgentState) -> dict:
    """Score tasks against knowledge graphs. Phase-aware difficulty. DRS augmentation."""
    logger.info("Executing Rerank_and_Retrieve node...")
    try:
        from orchestrator.discourse import DRS
        from orchestrator.phase_priors import (
            execution_phase, prior_difficulty,
        )
        from orchestrator.difficulty import compression_ratio
        from orchestrator.composition import merge_declared_edges

        memory = TriGraphMemory.load()
        dag = state["dag"]

        drs_dict = state.get("discourse_state") or {}
        drs = DRS.from_dict(drs_dict) if drs_dict else DRS()

        pg_has_nodes = (
            memory.semantic.number_of_nodes() > 0
            or memory.procedural.number_of_nodes() > 0
        )

        all_reranker_edges: list[ScoredEdge] = []
        phase = execution_phase(state.get("tasks_completed_count", 0))

        if dag and pg_has_nodes:
            for task in dag.get("tasks", []):
                raw_desc = task.get("description", "")
                augmented_desc = drs.augment_description(raw_desc)

                try:
                    retriever = memory._ensure_property_graph().as_retriever(
                        similarity_top_k=15
                    )
                    retrieved_nodes = retriever.retrieve(augmented_desc)
                except Exception as retrieval_exc:
                    logger.warning(
                        "Retrieval failed for task %s (%s).",
                        task.get("id"), retrieval_exc,
                    )
                    continue

                candidates = [
                    {
                        "id": node.metadata.get("node_id"),
                        "graph": node.metadata.get("graph_type", "unknown"),
                        "description": node.text,
                    }
                    for node in retrieved_nodes
                    if node.metadata.get("node_id")
                ]

                if candidates:
                    try:
                        edges = rerank_candidates(
                            task.get("id"), augmented_desc, candidates
                        )
                        all_reranker_edges.extend(edges)
                        for edge in edges:
                            graph_type = next(
                                (c["graph"] for c in candidates
                                 if c["id"] == edge.candidate_id),
                                "unknown",
                            )
                            memory.add_episodic_edge(
                                task.get("id"), edge.candidate_id,
                                {"graph": graph_type, "score": edge.score},
                            )
                    except Exception as rerank_exc:
                        logger.warning(
                            "Reranking failed for %s (%s).",
                            task.get("id"), rerank_exc,
                        )

        memory.save()

        task_difficulties = {}
        if dag:
            import networkx as nx
            from orchestrator.difficulty import compute_base_difficulty, propagate_difficulty

            task_list = dag.get("tasks", [])
            G = nx.DiGraph()
            for i, t in enumerate(task_list):
                G.add_node(t.get("id"))
                if i > 0:
                    G.add_edge(
                        task_list[i - 1].get("id"),
                        t.get("id"),
                        edge_type="observed",
                    )
            merge_declared_edges(G, task_list)

            if phase == 0:
                for t in task_list:
                    comp = compression_ratio(t.get("description", ""))
                    task_difficulties[t.get("id")] = prior_difficulty(
                        task_type=t.get("task_type", "transform"),
                        implementation_depth=t.get("implementation_depth", "library"),
                        actor_capability=t.get("actor_capability", "python"),
                        compression=comp,
                    )
            else:
                base = {
                    t.get("id"): compute_base_difficulty(
                        task_id=t.get("id"),
                        task_description=t.get("description", ""),
                        reranker_edges=all_reranker_edges,
                        composition_graph=G,
                    )
                    for t in task_list
                }
                task_difficulties = propagate_difficulty(G, base)

            for task_id, score in task_difficulties.items():
                if task_id in memory.episodic.nodes:
                    memory.episodic.nodes[task_id]["difficulty"] = score
            memory.save()

        return {
            "task_difficulties": task_difficulties,
            "messages": ["Scored tasks against knowledge graphs."],
        }

    except Exception as exc:
        logger.warning("Rerank_and_Retrieve failed (%s) — skipping.", exc)
        return {"messages": [f"Knowledge linking skipped ({type(exc).__name__})."]}


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
    """Generate code, execute in sandbox, record outcome and filesystem delta."""
    memory = TriGraphMemory.load()
    dag = state["dag"]
    idx = state["current_task_idx"]

    if not dag or idx >= len(dag.get("tasks", [])):
        return {"final_output": {"nodes": dag.get("tasks", []) if dag else [], "graphs": memory.to_dict()}}

    task = dag.get("tasks", [])[idx]

    logger.info("Executing Task: %s (attempt %d)", task.get("id"), state["current_attempt"])
    if task.get("id") in memory.episodic.nodes:
        memory.episodic.nodes[task.get("id")]["status"] = "running"

    # Get difficulty tier for model routing
    task_id = task.get("id", "")
    difficulties = state.get("task_difficulties", {})
    diff_score = difficulties.get(task_id, 0.5)
    from orchestrator.difficulty import difficulty_to_tier
    tier = difficulty_to_tier(diff_score)
    logger.info("Task %s difficulty=%.2f tier=%s", task_id, diff_score, tier)

    try:
        graph_context = memory.build_context(task_id)
    except Exception as ctx_exc:
        logger.warning("build_context failed for %s (%s) — using empty context.", task_id, ctx_exc)
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
    text = generate(context, task.get("description", ""), temperature=temp, tier=tier)

    code_to_run = text
    if tool_hint in ("python", "bash"):
        pattern = rf"```{tool_hint}\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code_to_run = match.group(1)

    update: dict = {}
    if tool_hint not in ("python", "bash"):
        memory.episodic.nodes.get(task_id, {}).update(status="success", response=text[:2000])
        update["dag"] = _with_task_status(dag, idx, "success")
        update["sandbox_results"] = [{"stdout": text[:1000], "stderr": "", "exit_code": 0, "code": "", "task_description": task.get("description", ""), "tool_hint": tool_hint}]
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1
        update["next_route"] = "Execute" if idx + 1 < len(dag.get("tasks", [])) else "Collect_Outputs"
        update["codebase_context"] = _append_workspace_entry(
            state.get("codebase_context", ""), task.get("description", ""), tool_hint, text[:300], "", 0
        )
    else:
        tool = ModalSandboxTool()

        # --- Execution-grounded composition: capture before/after state ---
        from orchestrator.composition import WorkspaceState, CompositionLedger
        composition_deltas = dict(state.get("composition_deltas", {}))

        try:
            before = WorkspaceState.capture(tool)
        except Exception:
            before = WorkspaceState.empty()

        raw_result = tool.forward(code=code_to_run, language=tool_hint)
        result = SandboxResult(
            stdout=raw_result["stdout"],
            stderr=raw_result["stderr"],
            exit_code=raw_result["exit_code"],
        )

        delta = None
        try:
            after = WorkspaceState.capture(tool)
            delta = after.diff(before)
            composition_deltas[task_id] = delta.to_dict()
        except Exception as comp_exc:
            logger.warning("Composition capture failed for %s (%s)", task_id, comp_exc)

        update["composition_deltas"] = composition_deltas
        update["sandbox_results"] = [{
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "code": code_to_run[:2000],
            "task_description": task.get("description", ""),
            "tool_hint": tool_hint,
        }]
        update["codebase_context"] = _append_workspace_entry(
            state.get("codebase_context", ""),
            task.get("description", ""),
            tool_hint,
            result.stdout,
            result.stderr,
            result.exit_code,
        )

        if result.success:
            memory.episodic.nodes.get(task_id, {}).update(status="success", output=result.stdout[:2000])
            update["dag"] = _with_task_status(dag, idx, "success")
            update["current_task_idx"] = idx + 1
            update["current_attempt"] = 1
            is_last = idx + 1 >= len(dag.get("tasks", []))
            update["next_route"] = "Collect_Outputs" if is_last else "Execute"

            # DRS: introduce files created by this task into the discourse scope
            if delta is not None:
                try:
                    from orchestrator.discourse import DRS
                    drs_dict = state.get("discourse_state") or {}
                    drs = DRS.from_dict(drs_dict) if drs_dict else DRS(label="main")
                    drs.introduce_from_delta(task_id, delta)
                    update["discourse_state"] = drs.to_dict()
                except Exception as drt_exc:
                    logger.debug("DRS update failed (%s) — skipping.", drt_exc)

            # DRS: commit retry scope if this was a retry attempt
            if state.get("current_attempt", 1) > 1:
                try:
                    from orchestrator.discourse import DRS
                    outer_drs_dict = state.get("discourse_state") or {}
                    retry_drs_dict = state.get("retry_discourse_state") or {}
                    if retry_drs_dict and outer_drs_dict:
                        outer_drs = DRS.from_dict(outer_drs_dict)
                        outer_drs.commit_retry_scope(retry_drs_dict)
                        update["discourse_state"] = outer_drs.to_dict()
                    update["retry_discourse_state"] = {}
                except Exception as commit_exc:
                    logger.debug("DRS retry commit failed (%s) — skipping.", commit_exc)

            # Phase 0: seed the procedural graph from this task's typed fields
            try:
                from orchestrator.phase_priors import execution_phase, seed_graph_from_task
                ph = execution_phase(state.get("tasks_completed_count", 0))
                if ph == 0:
                    seed_graph_from_task(task, memory)
            except Exception:
                pass

            # Increment completed task count
            update["tasks_completed_count"] = state.get("tasks_completed_count", 0) + 1
        else:
            fail_id = f"{task_id}_fail_{state['current_attempt']}"
            memory.add_episodic_node(fail_id, {
                "type": "FailedExecution", "stderr": result.stderr[:2000],
                "exit_code": result.exit_code, "attempt": state['current_attempt'],
            })
            memory.add_episodic_edge(task_id, fail_id, {"relation": "failed_execution"})
            update["dag"] = _with_task_status(dag, idx, "failed")
            update["next_route"] = "Handle_Failure"
            logger.warning("Task %s attempt %d failed (exit %d)", task_id, state['current_attempt'], result.exit_code)

    memory.save()
    return update


def handle_failure_node(state: AgentState) -> dict:
    """Handle execution failures. Opens a DRS retry scope. Escalates tier."""
    logger.info("Executing Handle_Failure node...")
    dag = state["dag"]
    idx = state["current_task_idx"]
    task = dag.get("tasks", [])[idx]
    memory = TriGraphMemory.load()

    last_stderr = ""
    last_exit_code = 1
    sandbox_results = state.get("sandbox_results", [])
    if sandbox_results:
        last = sandbox_results[-1]
        last_stderr = (
            last.get("stderr", "") if isinstance(last, dict)
            else getattr(last, "stderr", "")
        )
        last_exit_code = (
            last.get("exit_code", 1) if isinstance(last, dict)
            else getattr(last, "exit_code", 1)
        )

    new_attempt = state["current_attempt"] + 1
    task_id = task.get("id", "")
    diff_score = state.get("task_difficulties", {}).get(task_id, 0.5)

    update: dict = {
        "current_attempt": new_attempt,
        "messages": [
            f"Retrying task (attempt {new_attempt} of {MAX_RETRIES})...\n"
            f"Stderr: {last_stderr[:500] if last_stderr else '(empty)'}\n"
            f"Exit code: {last_exit_code}"
        ],
    }

    if new_attempt > MAX_RETRIES:
        # Retry budget exhausted — discard retry scope, mark failed
        try:
            from orchestrator.discourse import DRS
            outer_drs_dict = state.get("discourse_state") or {}
            if outer_drs_dict:
                outer_drs = DRS.from_dict(outer_drs_dict)
                outer_drs.discard_retry_scope()
                update["discourse_state"] = outer_drs.to_dict()
            update["retry_discourse_state"] = {}
        except Exception:
            pass

        if task_id in memory.episodic.nodes:
            memory.episodic.nodes[task_id]["status"] = "failed"
        tasks = [t.copy() for t in dag.get("tasks", [])]
        tasks[idx] = {**tasks[idx], "status": "failed"}
        update["dag"] = {**dag, "tasks": tasks}
        update["current_task_idx"] = idx + 1
        update["current_attempt"] = 1
        is_last = idx + 1 >= len(tasks)
        update["next_route"] = "Collect_Outputs" if is_last else "Execute"
        logger.warning("Task %s exhausted retries — marking failed.", task_id)
    else:
        # Open a DRS retry scope for the next attempt
        try:
            from orchestrator.discourse import DRS
            outer_drs_dict = state.get("discourse_state") or {}
            outer_drs = (
                DRS.from_dict(outer_drs_dict) if outer_drs_dict
                else DRS(label="main")
            )
            retry_drs = outer_drs.open_retry_scope(task_id)
            update["retry_discourse_state"] = retry_drs.to_dict()
        except Exception:
            pass

        # Tier escalation: one step up from current tier, or expert if
        # difficulty > 0.85. No causal model — the data does not support
        # one at this point (see assessment doc for full reasoning).
        from orchestrator.difficulty import difficulty_to_tier
        from orchestrator.phase_priors import prior_tier
        current_tier = difficulty_to_tier(diff_score)
        new_tier = prior_tier(diff_score, current_tier)
        logger.info(
            "Failure escalation: %s → %s (difficulty=%.2f)",
            current_tier, new_tier, diff_score,
        )
        tasks = [t.copy() for t in dag.get("tasks", [])]
        tasks[idx] = {**tasks[idx], "_retry_tier": new_tier}
        update["dag"] = {**dag, "tasks": tasks}
        update["next_route"] = "Execute"

    memory.save()
    return update


def collect_outputs_node(state: AgentState) -> dict:
    """Collect files from /workspace and include them in final_output.

    Runs find + reads key files (README, entry points, test results).
    Also derives the observed composition graph from the ledger and stores
    it in episodic memory.
    """
    logger.info("Collecting outputs from /workspace...")
    memory = TriGraphMemory.load()
    dag = state.get("dag")

    tool = ModalSandboxTool()
    workspace_files: dict[str, str] = {}

    try:
        # List all files in workspace
        list_result = tool.forward(
            code="find /workspace -type f | sort",
            language="bash",
            timeout=30,
        )
        file_list = [
            f.strip() for f in list_result.get("stdout", "").strip().splitlines()
            if f.strip()
        ]

        # Read key files (limit to first 20 to avoid blowing up the state)
        priority_patterns = [
            "README", "readme", "main", "app", "test_", "_test", "results", "output", "report"
        ]
        priority_files = []
        other_files = []
        for f in file_list:
            basename = os.path.basename(f).lower()
            if any(p.lower() in basename for p in priority_patterns):
                priority_files.append(f)
            else:
                other_files.append(f)

        files_to_read = (priority_files + other_files)[:20]
        for filepath in files_to_read:
            read_result = tool.forward(
                code=f"cat '{filepath}' 2>/dev/null | head -200",
                language="bash",
                timeout=15,
            )
            content = read_result.get("stdout", "").strip()
            if content:
                workspace_files[filepath] = content[:3000]

    except Exception as exc:
        logger.warning("Output collection failed (%s) — skipping file collection.", exc)

    # Derive observed composition graph from ledger and store in episodic memory
    try:
        from orchestrator.composition import CompositionLedger
        composition_deltas = state.get("composition_deltas", {})
        if composition_deltas:
            ledger = CompositionLedger.from_dict(composition_deltas)
            observed_graph = ledger.derive_dependency_graph()
            for src, dst in observed_graph.edges():
                memory.add_episodic_edge(src, dst, {"relation": "observed_composition"})
            memory.save()
    except Exception as comp_exc:
        logger.warning("Composition graph derivation failed (%s)", comp_exc)

    return {
        "final_output": {
            "nodes": dag.get("tasks", []) if dag else [],
            "graphs": memory.to_dict(),
            "workspace_files": workspace_files,
        },
        "messages": [f"Collected {len(workspace_files)} files from /workspace."],
        "next_route": "Validate",
    }


_VALIDATION_SYSTEM = """You are a software quality validator. Given:
- The original user request
- A list of completed tasks with their outputs
- Key files from the workspace

Evaluate whether the output satisfies the original request. Consider:
1. Does it address the core requirement?
2. Are there obvious errors or missing pieces?
3. Would a user be satisfied with this result?

Respond with a JSON object: {"satisfied": true/false, "issues": ["issue1", ...], "corrective_tasks": ["task description 1", ...]}
Only set corrective_tasks if satisfied is false."""


def validate_node(state: AgentState) -> dict:
    """Validate outputs and optionally create corrective tasks.

    Runs test suite if generated, checks for errors, uses LLM to evaluate
    whether the output satisfies the original prompt. If not satisfied and
    within MAX_ITERATIONS budget, creates corrective tasks.
    """
    logger.info("Validate node: evaluating output quality...")
    dag = state.get("dag")
    iteration = state.get("iteration_count", 0)

    if iteration >= MAX_ITERATIONS:
        logger.info("Reached MAX_ITERATIONS=%d — accepting current output.", MAX_ITERATIONS)
        return {
            "messages": [f"Validation complete (iteration limit {MAX_ITERATIONS} reached)."],
            "next_route": "__end__",
        }

    final_output = state.get("final_output", {})
    workspace_files = final_output.get("workspace_files", {})

    # Run test files if present
    tool = ModalSandboxTool()
    test_results = ""
    try:
        test_run = tool.forward(
            code=(
                "cd /workspace && "
                "if find . -name 'test_*.py' -o -name '*_test.py' | grep -q .; then "
                "  python -m pytest --tb=short -q 2>&1 | head -50; "
                "else "
                "  echo 'No test files found'; "
                "fi"
            ),
            language="bash",
            timeout=60,
        )
        test_results = test_run.get("stdout", "").strip()[:2000]
    except Exception as exc:
        test_results = f"[test runner error: {exc}]"

    # LLM evaluation
    file_summary = "\n".join(
        f"--- {path} ---\n{content[:500]}"
        for path, content in list(workspace_files.items())[:5]
    )
    task_summary = "\n".join(
        f"- {t.get('description', '')[:100]} [{t.get('status', 'unknown')}]"
        for t in (dag.get("tasks", []) if dag else [])
    )

    messages = [
        {"role": "system", "content": _VALIDATION_SYSTEM},
        {"role": "user", "content": (
            f"Original request: {state['user_prompt']}\n\n"
            f"Completed tasks:\n{task_summary}\n\n"
            f"Test results:\n{test_results or '(no tests run)'}\n\n"
            f"Key workspace files:\n{file_summary or '(no files found)'}"
        )},
    ]

    class ValidationResult(BaseModel):
        satisfied: bool
        issues: list[str] = []
        corrective_tasks: list[str] = []

    from pydantic import BaseModel
    try:
        result: ValidationResult = chat_completion(
            messages, temperature=0.2, max_tokens=1024, schema=ValidationResult
        )
    except Exception as exc:
        logger.warning("Validation LLM call failed (%s) — accepting output.", exc)
        return {
            "messages": ["Validation check failed — accepting current output."],
            "next_route": "__end__",
        }

    if result.satisfied:
        logger.info("Validation satisfied — output accepted.")
        return {
            "messages": [f"✓ Validation passed (iteration {iteration + 1})."],
            "next_route": "__end__",
        }

    logger.info("Validation not satisfied (issues: %s) — creating corrective tasks.", result.issues)

    if not result.corrective_tasks:
        return {
            "messages": ["Validation: output not fully satisfying but no corrective tasks suggested — accepting."],
            "next_route": "__end__",
        }

    # Create corrective tasks and re-enter execution
    corrective_nodes = [
        AtomicTaskNode(
            description=desc,
            tool_hint="python" if any(kw in desc.lower() for kw in ("python", "script", "implement", "write")) else "bash",
        )
        for desc in result.corrective_tasks
    ]

    existing_tasks = [AtomicTaskNode(**t) for t in dag.get("tasks", [])] if dag else []
    all_tasks = existing_tasks + corrective_nodes

    new_dag = TaskDAG(
        user_prompt=state["user_prompt"],
        intent_observation=f"Corrective iteration {iteration + 1}: {'; '.join(result.issues[:2])}",
        tasks=all_tasks,
    )

    memory = TriGraphMemory.load()
    for task in corrective_nodes:
        memory.add_episodic_node(task.id, task.model_dump())
    memory.save()

    return {
        "dag": new_dag.model_dump(),
        "current_task_idx": len(existing_tasks),  # Start at first corrective task
        "current_attempt": 1,
        "iteration_count": iteration + 1,
        "messages": [
            f"Validation iteration {iteration + 1}: {len(corrective_nodes)} corrective tasks created. "
            f"Issues: {'; '.join(result.issues[:3])}"
        ],
        "next_route": "Execute",
    }


def routing_after_execute(state: AgentState) -> str:
    """Determine the next step based on execution outcome."""
    route = state.get("next_route") or "Collect_Outputs"
    dag = state.get("dag")
    idx = state.get("current_task_idx", 0)
    if route == "Handle_Failure":
        return "Handle_Failure"
    if route == "Collect_Outputs" or not dag or idx >= len(dag.get("tasks", [])):
        return "Collect_Outputs"
    return "Execute"


def routing_after_failure(state: AgentState) -> str:
    dag = state.get("dag")
    idx = state.get("current_task_idx", 0)
    route = state.get("next_route", "Execute")
    if route == "Collect_Outputs" or not dag or idx >= len(dag.get("tasks", [])):
        return "Collect_Outputs"
    return "Execute"


def routing_after_validate(state: AgentState) -> str:
    route = state.get("next_route", "__end__")
    if route == "Execute":
        return "Execute"
    return END


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
    if intent == "RESEARCH":
        return "Research"
    return "Retrieve_Codebase"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # --- Semantic supervisor layer ---
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Conversation", conversation_node)
    workflow.add_node("Direct_Execute", direct_execute_node)
    workflow.add_node("Research", lambda s: __import__("orchestrator.web_research", fromlist=["research_node"]).research_node(s))
    workflow.add_node("Retrieve_Codebase", retrieve_codebase_node)

    # --- Execution pipeline ---
    workflow.add_node("Decompose", decompose_node)
    workflow.add_node("Rerank_and_Retrieve", rerank_and_retrieve_node)
    workflow.add_node("Execute", execute_node)
    workflow.add_node("Handle_Failure", handle_failure_node)
    workflow.add_node("Collect_Outputs", collect_outputs_node)
    workflow.add_node("Validate", validate_node)

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
            "Research": "Research",
        },
    )

    # Conversation → END
    workflow.add_edge("Conversation", END)

    # Direct_Execute → Execute
    workflow.add_edge("Direct_Execute", "Execute")

    # Research → Decompose (research results are in codebase_context)
    workflow.add_edge("Research", "Decompose")

    # RAG → Decompose → existing pipeline
    workflow.add_edge("Retrieve_Codebase", "Decompose")
    workflow.add_edge("Decompose", "Rerank_and_Retrieve")
    workflow.add_edge("Rerank_and_Retrieve", "Execute")

    workflow.add_conditional_edges(
        "Execute",
        routing_after_execute,
        {
            "Execute": "Execute",
            "Handle_Failure": "Handle_Failure",
            "Collect_Outputs": "Collect_Outputs",
        },
    )
    workflow.add_conditional_edges(
        "Handle_Failure",
        routing_after_failure,
        {
            "Execute": "Execute",
            "Collect_Outputs": "Collect_Outputs",
        },
    )
    workflow.add_edge("Collect_Outputs", "Validate")
    workflow.add_conditional_edges(
        "Validate",
        routing_after_validate,
        {
            "Execute": "Execute",
            END: END,
        },
    )

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
# Public Entrypoint — async generator (LangGraph 1.0 astream)
# ---------------------------------------------------------------------------

async def agent_loop_stream(user_prompt: str) -> AsyncIterator[dict]:
    """Run the full agent loop via LangGraph and stream the steps out.

    Uses LangGraph's native astream() — no threading, no queue wrapper.
    """
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
        "composition_deltas": {},
        "task_difficulties": {},
        "iteration_count": 0,
        "discourse_state": {},
        "retry_discourse_state": {},
        "tasks_completed_count": 0,
    }

    graph = get_compiled_graph()
    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}

    async for s in graph.astream(initial_state, config=config):
        node_name = list(s.keys())[0]
        s_update = s[node_name]
        full_state = graph.get_state(config).values
        safe_state = {
            "user_prompt": full_state.get("user_prompt", user_prompt),
            "messages": full_state.get("messages", []),
            "dag": s_update.get("dag") if "dag" in s_update else full_state.get("dag"),
            "intent": s_update.get("intent") if "intent" in s_update else full_state.get("intent"),
            "codebase_context": s_update.get("codebase_context") if "codebase_context" in s_update else full_state.get("codebase_context", ""),
            "current_task_idx": s_update.get("current_task_idx", full_state.get("current_task_idx", 0)),
            "current_attempt": s_update.get("current_attempt", full_state.get("current_attempt", 1)),
            "sandbox_results": full_state.get("sandbox_results", []),
            "final_output": s_update.get("final_output") if "final_output" in s_update else full_state.get("final_output"),
            "next_route": s_update.get("next_route", full_state.get("next_route")),
            "task_difficulties": full_state.get("task_difficulties", {}),
            "iteration_count": full_state.get("iteration_count", 0),
            "discourse_state": full_state.get("discourse_state", {}),
            "retry_discourse_state": full_state.get("retry_discourse_state", {}),
            "tasks_completed_count": full_state.get("tasks_completed_count", 0),
        }
        memory = TriGraphMemory.load()
        yield {
            "step": node_name,
            "state_snapshot": safe_state,
            "graphs": memory.to_dict(),
            "node_update": s_update,
        }


def agent_loop(user_prompt: str) -> dict[str, Any]:
    """Run the full agent loop synchronously and return the final graph state."""
    import asyncio

    async def _run():
        final_output = None
        async for update in agent_loop_stream(user_prompt):
            if update["state_snapshot"].get("final_output"):
                final_output = update["state_snapshot"]["final_output"]
        return final_output

    final_output = asyncio.run(_run())

    if final_output:
        return final_output

    memory = TriGraphMemory.load()
    return {"nodes": [], "graphs": memory.to_dict()}
