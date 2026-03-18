import operator
import os
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END

from orchestrator.tool_sandbox import execute_code
from orchestrator.task_parser import parse_prompt
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS
from orchestrator.llm_client import chat_completion
from orchestrator.graph_memory import TriGraphMemory

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict  # Persisted DRT mental model
    iteration: int

def decompose_and_evaluate(state: AgentState):
    """Montague Translation node: Map NL to formal logic frames."""
    print("🤖 [Jules] Decomposing request...")
    dag = parse_prompt(state["user_prompt"])
    
    # Initialize the primary Discourse box (DRS)
    drs = DRS(label="main")
    
    return {
        "dag": dag.model_dump(),
        "drs": drs.to_dict(),
        "messages": [{"role": "assistant", "content": f"Structured logic into {len(dag.tasks)} formal frames."}]
    }

def execute_tasks(state: AgentState):
    """Honest Neurosymbolic Execution: MLTT checks + Frege feedback loop."""
    from orchestrator import tool_sandbox 
    
    # Restore the Discourse mental model and Frege ledger
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger = CompositionLedger()
    
    tasks = state.get("dag", {}).get("tasks", [])
    overall_content = "### Formal Execution Phase\n\n"

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id")
        desc = task.get("description", "")
        
        # 1. DRT Anaphora Resolution: Augment description with accessible referents
        augmented_desc = drs.augment_description(desc)
        print(f"   ⏳ Task {i}: {augmented_desc[:80]}...")

        # 2. State Capture (Frege Start)
        before_state = WorkspaceState.capture(tool_sandbox)

        # 3. Execution (Montague -> Reality)
        # We pass the discourse context so the LLM 'knows' what it can see
        prompt = f"Task Description: {augmented_desc}\n\nPerform this action in the sandbox."
        res = chat_completion([{"role": "user", "content": prompt}])
        exec_out = execute_code(res)
        
        # 4. State Capture (Frege End)
        after_state = WorkspaceState.capture(tool_sandbox)
        
        # 5. Theoretical Update Loop
        delta = after_state.diff(before_state)
        ledger.record(task_id, before_state, after_state)
        drs.introduce_from_delta(task_id, delta) # Frege side-effects inform Discourse beliefs
        
        overall_content += f"#### Task {i}\n**Result:**\n{exec_out}\n\n"

    return {
        "drs": drs.to_dict(),
        "messages": [{"role": "assistant", "content": overall_content}]
    }

# --- LangGraph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_tasks", execute_tasks)

workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_tasks")
workflow.add_edge("execute_tasks", END)

agent_executor = workflow.compile()

async def agent_loop_stream(prompt: str):
    """The critical missing function required by core_app.py."""
    initial_state = {
        "messages": [{"role": "user", "content": prompt}], 
        "user_prompt": prompt,
        "iteration": 0
    }
    
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {
                "step": node,
                "state_snapshot": values,
                "node_update": values,
                "graphs": mem.to_dict()
            }
