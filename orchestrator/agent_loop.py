import operator
import re
import asyncio
from typing import Annotated, List, TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
import networkx as nx

from orchestrator.task_parser import parse_prompt
from orchestrator.llm_client import chat_completion
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.supervisor import supervisor_node, conversation_node, direct_execute_node
from orchestrator.web_research import research_node
from orchestrator.codebase_rag import retrieve_codebase_node
from orchestrator.rerank_engine import rerank_candidates, ScoredEdge
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS
from orchestrator.difficulty import compute_base_difficulty, difficulty_to_tier
import orchestrator.tool_sandbox as sandbox_tool

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    intent: str
    codebase_context: str
    dag: dict
    sandbox_results: Annotated[List[dict], operator.add]
    current_task_idx: int
    current_attempt: int
    reranker_edges: List[dict]
    composition_ledger: dict
    drs_dict: dict
    iteration: int

def decompose_node(state: AgentState) -> dict:
    ctx = state.get("codebase_context", "")
    dag = parse_prompt(state["user_prompt"], codebase_context=ctx)

    memory = TriGraphMemory.load()
    tasks = dag.tasks
    for t in tasks:
        memory.add_episodic_node(t.id, t.model_dump())
    memory.save()

    return {
        "dag": dag.model_dump(),
        "messages": [{"role": "assistant", "content": f"Decomposed into {len(tasks)} formal frames."}]
    }

def rerank_and_retrieve_node(state: AgentState) -> dict:
    dag = state.get("dag", {})
    tasks = dag.get("tasks", [])
    if not tasks:
        return {"messages": [{"role": "assistant", "content": "[RERANK] No tasks to rerank."}]}

    memory = TriGraphMemory.load()
    candidates = []

    for node, data in memory.semantic.nodes(data=True):
        candidates.append({"id": node, "description": data.get("content", str(node))})
    for node, data in memory.procedural.nodes(data=True):
        candidates.append({"id": node, "description": data.get("content", str(node))})

    all_edges = []
    if candidates:
        for t in tasks:
            tid = t.get("id")
            desc = t.get("description", "")
            try:
                edges = rerank_candidates(tid, desc, candidates)
                all_edges.extend([e.model_dump() for e in edges])
            except Exception:
                pass

    return {
        "reranker_edges": all_edges,
        "messages": [{"role": "assistant", "content": f"Scored {len(tasks)} tasks against knowledge graph."}]
    }

def execute_node(state: AgentState) -> dict:
    dag = state.get("dag", {})
    tasks = dag.get("tasks", [])
    idx = state.get("current_task_idx", 0)

    if idx >= len(tasks):
        return {"messages": [{"role": "assistant", "content": "[EXECUTE] All tasks completed."}]}

    task = tasks[idx]
    task_id = task.get("id", "unknown")
    desc = task.get("description", "")
    tool_hint = task.get("tool_hint", "python")

    memory = TriGraphMemory.load()

    edges_dicts = state.get("reranker_edges", [])
    edges = [ScoredEdge(**e) for e in edges_dicts]

    ledger_dict = state.get("composition_ledger", {})
    ledger = CompositionLedger.from_dict(ledger_dict)
    try:
        comp_graph = ledger.derive_dependency_graph()
    except Exception:
        comp_graph = nx.DiGraph()
        
    diff_score = compute_base_difficulty(task_id, desc, edges, comp_graph, target_code="")
    tier = difficulty_to_tier(diff_score)

    if task_id in memory.episodic:
        memory.episodic.nodes[task_id]["difficulty"] = diff_score

    context = memory.build_context(task_id)
    drs_dict = state.get("drs_dict", {})
    drs = DRS.from_dict(drs_dict)

    augmented_desc = drs.augment_description(desc)

    prompt = f"Task: {augmented_desc}\nContext:\n{context}\n\nWrite {tool_hint} code to accomplish this task. Provide ONLY the executable code inside standard markdown blocks."

    temp = 0.3 if tool_hint in ("python", "bash") else 0.7

    try:
        res = chat_completion([{"role": "user", "content": prompt}], tier=tier, temperature=temp)
        code_text = res.choices[0].message.content if hasattr(res, 'choices') else str(res)
    except Exception as e:
        code_text = f"Error calling LLM: {str(e)}"

    match = re.search(f"```(?:{tool_hint}|python|bash|sh)\n(.*?)\n```", code_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
    else:
        extracted_code = code_text.strip()
        if extracted_code.startswith("```"):
            extracted_code = "\n".join(extracted_code.split("\n")[1:-1])
        
    before = WorkspaceState.capture(sandbox_tool)

    if "Error calling LLM" in extracted_code:
        exec_out = {"stdout": "", "stderr": extracted_code, "exit_code": 1}
    else:
        exec_res = sandbox_tool.forward(extracted_code, language=tool_hint)
        exec_out = {
            "stdout": exec_res.get("stdout", ""),
            "stderr": exec_res.get("stderr", exec_res.get("error", "")),
        }
        stdout_str = str(exec_out["stdout"])
        exec_out["exit_code"] = 1 if "Error:" in stdout_str or "Exception" in stdout_str else 0

    after = WorkspaceState.capture(sandbox_tool)

    ledger.record(task_id, before, after)

    delta = after.diff(before)
    drs.introduce_from_delta(task_id, delta)

    status = "success" if exec_out["exit_code"] == 0 else "failed"
    if task_id not in memory.episodic:
        memory.add_episodic_node(task_id, {"description": desc, "tool_hint": tool_hint, "status": status})
    else:
        memory.episodic.nodes[task_id]["status"] = status
    memory.save()

    return {
        "sandbox_results": [{
            "task_id": task_id,
            "task_description": desc,
            "code": extracted_code,
            "stdout": exec_out.get("stdout", ""),
            "stderr": exec_out.get("stderr", ""),
            "exit_code": exec_out["exit_code"],
            "tool_hint": tool_hint
        }],
        "composition_ledger": ledger.to_dict(),
        "drs_dict": drs.to_dict(),
        "messages": [{"role": "assistant", "content": f"Task executed on `{tier}` tier: {desc[:50]}..."}]
    }

def validate_node(state: AgentState) -> dict:
    results = state.get("sandbox_results", [])
    if not results:
        return {"messages": [{"role": "assistant", "content": "[VALIDATE] No results to validate."}]}
    latest = results[-1]
    exit_code = latest.get("exit_code", 0)
    stdout = latest.get("stdout", "")

    if exit_code != 0 or "Error:" in str(stdout):
        return {"messages": [{"role": "assistant", "content": "[VALIDATE] Execution failed."}]}
    else:
        return {"messages": [{"role": "assistant", "content": "[VALIDATE] Execution succeeded."}]}

def should_retry(state: AgentState) -> str:
    results = state.get("sandbox_results", [])
    if not results:
        return "Next_Task"
    latest = results[-1]
    exit_code = latest.get("exit_code", 0)
    stdout = latest.get("stdout", "")

    if exit_code != 0 or "Error:" in str(stdout):
        if state.get("current_attempt", 1) < 2:
            return "Handle_Failure"
        else:
            return "Next_Task"
    return "Next_Task"

def handle_failure_node(state: AgentState) -> dict:
    attempt = state.get("current_attempt", 1)
    return {
        "current_attempt": attempt + 1,
        "messages": [{"role": "assistant", "content": f"[FAILURE] Retrying... (Attempt {attempt + 1})"}]
    }

def next_task_node(state: AgentState) -> dict:
    idx = state.get("current_task_idx", 0)
    return {
        "current_task_idx": idx + 1,
        "current_attempt": 1,
        "messages": [{"role": "assistant", "content": f"[NEXT] Moving to task {idx + 2}."}]
    }

def check_next_task(state: AgentState) -> str:
    dag = state.get("dag", {})
    tasks = dag.get("tasks", [])
    idx = state.get("current_task_idx", 0)
    if idx < len(tasks):
        return "Execute"
    return "Collect_Outputs"

def collect_outputs_node(state: AgentState) -> dict:
    results = state.get("sandbox_results", [])
    return {
        "messages": [{"role": "assistant", "content": f"[COLLECT] Pipeline complete with {len(results)} execution steps."}]
    }

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Conversation", conversation_node)
workflow.add_node("Research", research_node)
workflow.add_node("Retrieve_Codebase", retrieve_codebase_node)
workflow.add_node("Decompose", decompose_node)
workflow.add_node("Direct_Execute", direct_execute_node)
workflow.add_node("Rerank_and_Retrieve", rerank_and_retrieve_node)
workflow.add_node("Execute", execute_node)
workflow.add_node("Validate", validate_node)
workflow.add_node("Handle_Failure", handle_failure_node)
workflow.add_node("Next_Task", next_task_node)
workflow.add_node("Collect_Outputs", collect_outputs_node)

workflow.set_entry_point("Supervisor")

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "DECOMPOSE")
    if intent == "CONVERSATION": return "Conversation"
    if intent == "DIRECT_EXECUTE": return "Direct_Execute"
    if intent == "RESEARCH": return "Research"
    return "Retrieve_Codebase"

workflow.add_conditional_edges("Supervisor", route_intent)

workflow.add_edge("Conversation", END)
workflow.add_edge("Research", "Decompose")
workflow.add_edge("Retrieve_Codebase", "Decompose")
workflow.add_edge("Decompose", "Rerank_and_Retrieve")
workflow.add_edge("Direct_Execute", "Rerank_and_Retrieve")

workflow.add_edge("Rerank_and_Retrieve", "Execute")

workflow.add_edge("Execute", "Validate")

# Only 1 conditional edge for Validate
workflow.add_conditional_edges("Validate", should_retry, {"Next_Task": "Next_Task", "Handle_Failure": "Handle_Failure"})

workflow.add_edge("Handle_Failure", "Execute")

def next_task_router(state: AgentState) -> str:
    return check_next_task(state)

# To properly route Next_Task, we add conditional edges
workflow.add_conditional_edges("Next_Task", next_task_router, {"Execute": "Execute", "Collect_Outputs": "Collect_Outputs"})

workflow.add_edge("Collect_Outputs", END)

agent_executor = workflow.compile()

async def agent_loop_stream(prompt: str):
    """Provides the async stream expected by Chainlit in core_app.py"""
    initial_state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_prompt": prompt,
        "iteration": 0,
        "current_task_idx": 0,
        "current_attempt": 1,
        "sandbox_results": []
    }

    async_gen = agent_executor.astream(initial_state)

    while True:
        try:
            event = await asyncio.wait_for(async_gen.__anext__(), timeout=2.0)
            for node, values in event.items():
                mem = TriGraphMemory.load()
                yield {"step": node, "state_snapshot": values, "node_update": values, "graphs": mem.to_dict()}
        except asyncio.TimeoutError:
            yield {"step": "keep-alive", "state_snapshot": {}, "node_update": {}, "graphs": {}}
        except StopAsyncIteration:
            break

def agent_loop(prompt: str) -> dict:
    initial_state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_prompt": prompt,
        "iteration": 0,
        "current_task_idx": 0,
        "current_attempt": 1,
        "sandbox_results": []
    }
    return agent_executor.invoke(initial_state)
