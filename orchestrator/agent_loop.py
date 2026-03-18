import operator
import os
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
import networkx as nx

from orchestrator.tool_sandbox import execute_code
from orchestrator.task_parser import parse_prompt
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS
from orchestrator.llm_client import chat_completion, ModalKeepAlive
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.difficulty import compute_base_difficulty, difficulty_to_tier

# CRITICAL FIX: Using the actual functional entry points from your repository
from orchestrator.indexer import build_knowledge_graphs

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict
    current_task_idx: int
    ledger: dict

def decompose_and_evaluate(state: AgentState):
    """
    MONTAGUE GRAMMAR & FILLMORE FRAMES:
    Translates natural language into strict, typed operational frames (Transform, Query).
    """
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
    from orchestrator import tool_sandbox 
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task = tasks[idx]
    
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger = CompositionLedger()
    if state.get("ledger"): ledger.events = state["ledger"].get("events", [])

    # KEEP-ALIVE: Pulses the Modal container so it doesn't die during theoretical calculations
    with ModalKeepAlive(tier="moderate"):
        mem = TriGraphMemory.load()
        
        # 1. EPISTEMIC ASSESSMENT (RAG)
        # We use your graph_memory's native vector retriever, just like in codebase_rag.py
        try:
            retriever = mem.as_vector_retriever(similarity_top_k=5)
            knowledge_nodes = retriever.retrieve(task.get("description", ""))
        except Exception as e:
            print(f"      ⚠️ Retrieval failed (Graph may be empty): {e}")
            knowledge_nodes = []
        
        # 2. KOLMOGOROV & STRUCTURAL DIFFICULTY
        # Computes AST density (Kolmogorov proxy) and offsets it with Reranker Epistemic coverage
        score = compute_base_difficulty(
            task_id=task.get("id", ""),
            task_description=task.get("description", ""),
            reranker_edges=knowledge_nodes,
            composition_graph=nx.DiGraph(), 
            target_code=""
        )
        tier = difficulty_to_tier(score)
        print(f"   🧭 Routing Task {idx+1}/{len(tasks)} to [{tier.upper()}] (Score: {score:.2f})")

        # 3. DISCOURSE REPRESENTATION THEORY (DRT) & MLTT
        # Augments the prompt with specific anaphora resolution ("the file" -> "/workspace/app.py")
        desc = drs.augment_description(task.get("description", ""))
        
        prompt = f"Implement task. Context:\n{desc}\n\nRetrieved Knowledge:\n"
        prompt += "\n".join([n.text for n in knowledge_nodes[:2]]) if knowledge_nodes else "None"
        
        # 4. FREGE COMPOSITIONALITY (Pre-State)
        before_state = WorkspaceState.capture(tool_sandbox)
        
        # Generation Step
        res = chat_completion([{"role": "user", "content": prompt}], tier=tier)
        
    # --- Execute and Observe Side Effects (The Frege Ledger) ---
    exec_out = execute_code(str(res))
    after_state = WorkspaceState.capture(tool_sandbox)
    
    delta = after_state.diff(before_state)
    ledger.record(task.get("id"), before_state, after_state)
    drs.introduce_from_delta(task.get("id"), delta)
    
    # 5. CONTINUOUS EPISTEMIC FEEDBACK
    # Honest Repo Integration: If files changed, we run your native indexer
    if delta.created or delta.modified:
        print("   🧠 [Epistemic Update] Ingesting new artifacts into Tri-Graph...")
        # build_knowledge_graphs handles gitignore, chunking, and typed schema insertion
        build_knowledge_graphs("/workspace")
    
    return {
        "drs": drs.to_dict(),
        "current_task_idx": idx + 1,
        "ledger": ledger.to_dict(),
        "messages": [{"role": "assistant", "content": f"#### Task {idx+1} ({tier})\n**Result:**\n{exec_out}\n\n"}]
    }

def route_next_step(state: AgentState) -> str:
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    if idx < len(tasks): return "execute_single_task"
    return "end"

# --- Dynamic Graph Wiring ---
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_single_task", execute_single_task)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_single_task")
workflow.add_conditional_edges("execute_single_task", route_next_step, {"execute_single_task": "execute_single_task", "end": END})
agent_executor = workflow.compile()

async def agent_loop_stream(prompt: str):
    initial_state = {"messages": [{"role": "user", "content": prompt}], "user_prompt": prompt, "current_task_idx": 0}
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {"step": node, "state_snapshot": values, "node_update": values, "graphs": mem.to_dict()}
