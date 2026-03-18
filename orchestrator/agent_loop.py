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

# --- Ingestion Imports ---
from orchestrator.extractor import Extractor
from orchestrator.indexer import Indexer
from orchestrator.codebase_rag import RAGQueryEngine

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict
    current_task_idx: int  
    ledger: dict

def decompose_and_evaluate(state: AgentState):
    print("🤖 Decomposing request...")
    dag = parse_prompt(state["user_prompt"])
    return {
        "dag": dag.model_dump(),
        "drs": DRS(label="main").to_dict(),
        "current_task_idx": 0,
        "ledger": CompositionLedger().to_dict(),
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} formal frames."}]
    }

def ingest_delta_to_graph(delta, workspace_path="/workspace"):
    """THEORETICAL FIX: Epistemic Update Phase.
    Parses newly created/modified files and updates the Tri-Graph."""
    mem = TriGraphMemory.load()
    extractor = Extractor()
    indexer = Indexer(mem)
    
    files_to_process = set(delta.created) | set(delta.modified)
    
    if not files_to_process:
        return
        
    print(f"   🧠 [Epistemic Update] Ingesting {len(files_to_process)} new artifacts into Knowledge Graph...")
    for file_path in files_to_process:
        full_path = os.path.join(workspace_path, file_path.lstrip('/'))
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Extract AST/Symbols and index into semantic graph
                symbols = extractor.extract(content, file_path)
                indexer.index_symbols(symbols, file_path)
            except Exception as e:
                print(f"      ⚠️ Failed to ingest {file_path}: {e}")
                
    mem.save()

def execute_single_task(state: AgentState):
    from orchestrator import tool_sandbox 
    
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task = tasks[idx]
    
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger = CompositionLedger()
    if state.get("ledger"):
        ledger.events = state["ledger"].get("events", [])
    
    # 1. HOLD INFRASTRUCTURE WARM
    # The context manager continuously pings Modal while we calculate
    with ModalKeepAlive(tier="moderate"):
        
        # 2. THEORETICAL ASSESSMENT (Kolmogorov & RAG)
        mem = TriGraphMemory.load()
        rag_engine = RAGQueryEngine(mem)
        
        # Pull epistemic context from the updated RAG graph
        context_nodes = rag_engine.query(task.get("description", ""))
        
        difficulty_score = compute_base_difficulty(
            task_id=task.get("id", ""),
            task_description=task.get("description", ""), 
            reranker_edges=context_nodes, # Feeds RAG success into difficulty
            composition_graph=nx.DiGraph(), 
            target_code=""
        )
        assigned_tier = difficulty_to_tier(difficulty_score)
        print(f"   🧭 Routing Task {idx+1}/{len(tasks)} to Tier: [{assigned_tier.upper()}] (Score: {difficulty_score:.2f})")

        # 3. DRT Resolution & Frege Start
        desc = drs.augment_description(task.get("description", ""))
        
        # Inject the retrieved RAG context directly into the LLM prompt
        rag_context_str = "\n".join([n.text for n in context_nodes[:3]]) if context_nodes else "No prior knowledge."
        prompt = f"Implement this task.\n\nDiscourse Context:\n{desc}\n\nRetrieved Knowledge:\n{rag_context_str}"

    # --- Container lock released, safe to make the real generation call ---
    
    before_state = WorkspaceState.capture(tool_sandbox)
    
    # 4. EXECUTION
    res = chat_completion([{"role": "user", "content": prompt}], tier=assigned_tier)
    code_text = res.choices[0].message.content if hasattr(res, 'choices') else str(res)
    exec_out = execute_code(code_text)
    
    # 5. Frege Observation & DRT Update
    after_state = WorkspaceState.capture(tool_sandbox)
    delta = after_state.diff(before_state)
    ledger.record(task.get("id"), before_state, after_state)
    drs.introduce_from_delta(task.get("id"), delta) 
    
    # 6. EPISTEMIC UPDATE (Ingest new research/code)
    ingest_delta_to_graph(delta)
    
    msg = f"#### Task {idx+1} ({assigned_tier})\n**Result:**\n{exec_out}\n\n"
    
    return {
        "drs": drs.to_dict(),
        "current_task_idx": idx + 1,
        "ledger": ledger.to_dict(),
        "messages": [{"role": "assistant", "content": msg}]
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
