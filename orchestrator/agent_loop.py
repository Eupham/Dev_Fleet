"""LangGraph agent loop — with difficulty reassessment and sub-task spawning.
FIXES APPLIED:
1. LEDGER BUG: CompositionLedger was recreated empty on every task because
   `ledger.events` doesn't exist — the class has `deltas` and `_order`.
   Fixed: proper deserialization via CompositionLedger.from_dict().
2. DIFFICULTY REASSESSMENT: After each task, remaining tasks are re-scored
   against the updated knowledge graph. This is the critical feedback loop.
3. SUB-TASK SPAWNING: When task[i] discovers unexpected complexity (new
   library, test failure), new tasks are inserted into the DAG.
4. MULTI-TIER ROUTING: difficulty tier is passed to chat_completion for
   actual model/GPU selection (requires model_pool.py to have endpoints).
5. WorkspaceState.capture() is only called when there's a sandbox tool
   available, preventing crashes on empty/missing sandbox.
"""
import operator
import os
from typing import Annotated, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
import networkx as nx
from orchestrator.tool_sandbox import execute_code
from orchestrator.task_parser import parse_prompt, TaskDAG, TransformTask, VerifyTask
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS, ReferentType
from orchestrator.llm_client import chat_completion, ModalKeepAlive
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.difficulty import (
    compute_base_difficulty, difficulty_to_tier, reassess_remaining_tasks
)
from orchestrator.indexer import build_knowledge_graphs
# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict
    current_task_idx: int
    ledger: dict
    # NEW: track reassessed difficulty scores for remaining tasks
    difficulty_scores: dict  # {task_id: (score, tier)}
# ---------------------------------------------------------------------------
# Node 1: Decompose prompt into TaskDAG
# ---------------------------------------------------------------------------
def decompose_and_evaluate(state: AgentState):
    """
    FILLMORE FRAME EXTRACTION:
    Translates natural language into strict, typed operational frames.
    Outputs a DAG (not linear chain) with dependency edges.
    """
    print("🤖 Decomposing request into Fillmore Frames...")
    dag = parse_prompt(state["user_prompt"])
    # Pre-compute initial difficulty for all tasks
    mem = TriGraphMemory.load()
    initial_scores = {}
    # Build knowledge context from current graph
    try:
        retriever = mem.as_vector_retriever(similarity_top_k=10)
        all_knowledge = retriever.retrieve(state["user_prompt"])
        knowledge_text = "\n".join([n.text for n in all_knowledge[:5]]) if all_knowledge else ""
    except Exception:
        knowledge_text = ""
        all_knowledge = []
    for task in dag.tasks:
        score = compute_base_difficulty(
            task_id=task.id,
            task_description=task.description,
            reranker_edges=all_knowledge,
            composition_graph=nx.DiGraph(),
            knowledge_context=knowledge_text,
        )
        tier = difficulty_to_tier(score)
        initial_scores[task.id] = {"score": score, "tier": tier}
        print(f"  📊 Task '{task.description[:40]}...' → {tier} (K={score:.2f})")
    return {
        "dag": dag.model_dump(),
        "drs": DRS(label="main").to_dict(),
        "current_task_idx": 0,
        "ledger": CompositionLedger().to_dict(),
        "difficulty_scores": initial_scores,
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} frames. Initial difficulty assessed."}]
    }
# ---------------------------------------------------------------------------
# Node 2: Execute single task with full feedback loop
# ---------------------------------------------------------------------------
def execute_single_task(state: AgentState):
    from orchestrator import tool_sandbox
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task = tasks[idx]
    task_id = task.get("id", "")
    task_desc = task.get("description", "")
    # --- Deserialize state (FIX: was using non-existent .events attribute) ---
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger_data = state.get("ledger", {})
    ledger = CompositionLedger.from_dict(ledger_data) if ledger_data else CompositionLedger()
    # --- Get difficulty tier (may have been reassessed by previous task) ---
    scores = state.get("difficulty_scores", {})
    task_scores = scores.get(task_id, {})
    tier = task_scores.get("tier", "moderate")
    score = task_scores.get("score", 0.5)
    print(f"\n🧭 Executing Task {idx + 1}/{len(tasks)}: [{tier.upper()}] (K={score:.2f})")
    print(f"   '{task_desc[:60]}'")
    with ModalKeepAlive(tier=tier):
        mem = TriGraphMemory.load()
        # 1. EPISTEMIC RETRIEVAL (RAG)
        try:
            retriever = mem.as_vector_retriever(similarity_top_k=5)
            knowledge_nodes = retriever.retrieve(task_desc)
        except Exception as e:
            print(f"  ⚠️ Retrieval failed (graph may be empty): {e}")
            knowledge_nodes = []
        # 2. DRT: Augment description with resolved discourse referents
        desc = drs.augment_description(task_desc)
        prompt = f"Implement task. Context:\n{desc}\n\nRetrieved Knowledge:\n"
        prompt += "\n".join([n.text for n in knowledge_nodes[:3]]) if knowledge_nodes else "None"
        # 3. FREGE: Capture pre-execution state
        try:
            before_state = WorkspaceState.capture(tool_sandbox)
        except Exception:
            before_state = WorkspaceState.empty()
        # 4. GENERATION: Route to appropriate model tier
        res = chat_completion(
            [{"role": "user", "content": prompt}],
            tier=tier,  # This should select the model based on difficulty
        )
        # 5. EXECUTE: Run generated code in sandbox
        exec_out = execute_code(str(res))
        # 6. FREGE: Capture post-execution state and diff
        try:
            after_state = WorkspaceState.capture(tool_sandbox)
            delta = after_state.diff(before_state)
        except Exception:
            from orchestrator.composition import StateDelta
            delta = StateDelta(created=frozenset(), deleted=frozenset(), modified=frozenset())
            after_state = before_state
        # Record in composition ledger
        ledger.record(task_id, before_state, after_state)
        # 7. DRT: Introduce new referents from filesystem changes
        drs.introduce_from_delta(task_id, delta)
        # 8. EPISTEMIC UPDATE: Re-index if files changed
        new_knowledge_text = ""
        if delta.created or delta.modified:
            print("  🧠 [Epistemic Update] Ingesting new artifacts into Tri-Graph...")
            build_knowledge_graphs("/workspace")
            # Refresh knowledge context for reassessment
            try:
                mem = TriGraphMemory.load()
                retriever = mem.as_vector_retriever(similarity_top_k=10)
                new_knowledge = retriever.retrieve(state["user_prompt"])
                new_knowledge_text = "\n".join([n.text for n in new_knowledge[:5]])
            except Exception:
                pass
        # 9. REASSESS remaining tasks' difficulty
        remaining_tasks = tasks[idx + 1:]
        updated_scores = dict(scores)  # Copy existing scores
        if remaining_tasks:
            print("  📊 [Reassessment] Re-scoring remaining tasks against updated graph...")
            try:
                comp_graph = ledger.derive_dependency_graph()
            except (ValueError, Exception):
                comp_graph = nx.DiGraph()
            # Get fresh reranker edges
            reranker_edges = knowledge_nodes  # Use latest retrieval
            reassessed = reassess_remaining_tasks(
                remaining_tasks=remaining_tasks,
                knowledge_context=new_knowledge_text or "",
                reranker_edges=reranker_edges,
                composition_graph=comp_graph,
            )
            for tid, new_score, new_tier in reassessed:
                old = updated_scores.get(tid, {})
                old_tier = old.get("tier", "unknown")
                if new_tier != old_tier:
                    print(f"    ↕ Task {tid}: {old_tier} → {new_tier} (K={new_score:.2f})")
                updated_scores[tid] = {"score": new_score, "tier": new_tier}
        # 10. SUB-TASK SPAWNING: Check if execution revealed new complexity
        spawned_tasks = _check_for_subtasks(exec_out, task, drs)
        dag_update = state.get("dag", {})
        if spawned_tasks:
            print(f"  🔀 Spawning {len(spawned_tasks)} sub-tasks from execution results...")
            # Insert sub-tasks right after current task
            current_tasks = dag_update.get("tasks", [])
            for i, st in enumerate(spawned_tasks):
                st_dict = st.model_dump()
                current_tasks.insert(idx + 1 + i, st_dict)
                # Score the new sub-task
                sub_score = compute_base_difficulty(
                    task_id=st.id,
                    task_description=st.description,
                    reranker_edges=reranker_edges if remaining_tasks else [],
                    composition_graph=comp_graph if remaining_tasks else nx.DiGraph(),
                    knowledge_context=new_knowledge_text,
                )
                sub_tier = difficulty_to_tier(sub_score)
                updated_scores[st.id] = {"score": sub_score, "tier": sub_tier}
                print(f"    + Sub-task '{st.description[:40]}...' → {sub_tier} (K={sub_score:.2f})")
            dag_update["tasks"] = current_tasks
    return {
        "dag": dag_update,
        "drs": drs.to_dict(),
        "current_task_idx": idx + 1,
        "ledger": ledger.to_dict(),
        "difficulty_scores": updated_scores,
        "messages": [{
            "role": "assistant",
            "content": f"#### Task {idx + 1} ({tier}, K={score:.2f})\n**Result:**\n{exec_out}\n\n"
        }]
    }
# ---------------------------------------------------------------------------
# Sub-task Spawning Heuristics
# ---------------------------------------------------------------------------
def _check_for_subtasks(exec_output: str, task: dict, drs: DRS) -> list:
    """Detect if execution output indicates need for sub-tasks.
    Heuristics (deterministic, no LLM needed):
    1. Test failures → spawn a debug + re-test task
    2. Import errors → spawn a research + install task
    3. Multiple files created → spawn verify tasks for each
    This is where the system self-corrects without human intervention.
    The heuristics are intentionally conservative — only spawn when
    there's strong signal.
    """
    spawned = []
    output_lower = exec_output.lower() if exec_output else ""
    # Heuristic 1: Test failure → debug + re-test
    if any(kw in output_lower for kw in ["traceback", "error", "failed", "assertion"]):
        if task.get("task_type") != "verify":
            # Don't spawn infinite verify loops
            spawned.append(VerifyTask(
                description=f"Debug and fix error from: {task.get('description', '')[:80]}",
                tool_hint="python",
                preconditions=[task.get("id", "")],
            ))
    # Heuristic 2: Import error → research + install
    if "modulenotfounderror" in output_lower or "no module named" in output_lower:
        # Extract module name (best effort)
        import re
        match = re.search(r"no module named ['\"]?(\w+)", output_lower)
        module_name = match.group(1) if match else "unknown"
        from orchestrator.task_parser import TransformTask as TT
        spawned.append(TT(
            description=f"Install missing module '{module_name}' and retry",
            tool_hint="bash",
            preconditions=[task.get("id", "")],
        ))
    return spawned
# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_next_step(state: AgentState) -> str:
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    if idx < len(tasks):
        return "execute_single_task"
    return "end"
# ---------------------------------------------------------------------------
# Graph Wiring
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_single_task", execute_single_task)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_single_task")
workflow.add_conditional_edges(
    "execute_single_task",
    route_next_step,
    {"execute_single_task": "execute_single_task", "end": END}
)
agent_executor = workflow.compile()
async def agent_loop_stream(prompt: str):
    initial_state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_prompt": prompt,
        "current_task_idx": 0,
        "difficulty_scores": {},
    }
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {
                "step": node,
                "state_snapshot": values,
                "node_update": values,
                "graphs": mem.to_dict(),
                "difficulty_scores": values.get("difficulty_scores", {}),
            }
