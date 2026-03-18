import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
import networkx as nx

from orchestrator.tool_sandbox import execute_code
from orchestrator.task_parser import parse_prompt
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS
from orchestrator.llm_client import chat_completion, ping_tier
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.difficulty import compute_base_difficulty, difficulty_to_tier

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict
    current_task_idx: int  # Tracks DAG traversal
    ledger: dict

def decompose_and_evaluate(state: AgentState):
    """Montague Translation: Creates the DAG."""
    print("🤖 Decomposing request...")
    dag = parse_prompt(state["user_prompt"])
    
    return {
        "dag": dag.model_dump(),
        "drs": DRS(label="main").to_dict(),
        "current_task_idx": 0,
        "ledger": CompositionLedger().to_dict(),
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} formal frames."}]
    }

def execute_single_task(state: AgentState):
    """Executes a SINGLE task, reassessing context and routing dynamically."""
    from orchestrator import tool_sandbox 
    
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task = tasks[idx]
    
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger = CompositionLedger()
    if state.get("ledger"):
        ledger.events = state["ledger"].get("events", [])
    
    # 1. EMIT HEARTBEAT: Keep Modal warm while we calculate theory
    # Defaulting ping to moderate, you can track the active tier in state if desired
    ping_tier("moderate") 

    # 2. THEORETICAL ASSESSMENT (Kolmogorov & Epistemic)
    # Re-evaluate difficulty based on the *current* state of the sandbox
    mem = TriGraphMemory.load()
    current_code = "" # In production, extract the target file content here
    
    difficulty_score = compute_base_difficulty(
        task_id=task.get("id"),
        reranker_edges=[], # Populate via RAG queries against mem.semantic
        composition_graph=nx.DiGraph(), 
        target_code=current_code
    )
    assigned_tier = difficulty_to_tier(difficulty_score)
    print(f"   🧭 Routing Task {idx+1}/{len(tasks)} to Tier: [{assigned_tier.upper()}] (Score: {difficulty_score:.2f})")

    # 3. DRT Resolution & Frege Start
    desc = drs.augment_description(task.get("description", ""))
    before_state = WorkspaceState.capture(tool_sandbox)

    # 4. EXECUTION
    prompt = f"Implement this task. Discourse Context:\n{desc}"
    res = chat_completion([{"role": "user", "content": prompt}], tier=assigned_tier)
    code_text = res.choices[0].message.content if hasattr(res, 'choices') else str(res)
    exec_out = execute_code(code_text)
    
    # 5. Frege Observation & DRT Update
    after_state = WorkspaceState.capture(tool_sandbox)
    delta = after_state.diff(before_state)
    ledger.record(task.get("id"), before_state, after_state)
    drs.introduce_from_delta(task.get("id"), delta) 
    
    msg = f"#### Task {idx+1} ({assigned_tier})\n**Result:**\n{exec_out}\n\n"
    
    return {
        "drs": drs.to_dict(),
        "current_task_idx": idx + 1,
        "ledger": ledger.to_dict(),
        "messages": [{"role": "assistant", "content": msg}]
    }

def route_next_step(state: AgentState) -> str:
    """DAG Router: Checks if we have more tasks."""
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    if idx < len(tasks):
        return "execute_single_task"
    return "end"

# --- Dynamic Graph Wiring ---
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_single_task", execute_single_task)

workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_single_task")

# Conditional routing loops back to execution until all tasks are finished
workflow.add_conditional_edges(
    "execute_single_task",
    route_next_step,
    {
        "execute_single_task": "execute_single_task",
        "end": END
    }
)

agent_executor = workflow.compile()

async def agent_loop_stream(prompt: str):
    initial_state = {"messages": [{"role": "user", "content": prompt}], "user_prompt": prompt, "current_task_idx": 0}
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {"step": node, "state_snapshot": values, "node_update": values, "graphs": mem.to_dict()}
