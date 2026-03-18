import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END

from orchestrator.tool_sandbox import execute_code, execute_bash
from orchestrator.task_parser import parse_prompt
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS
from orchestrator.llm_client import chat_completion

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict  # Persisted DRT mental model

def decompose_and_evaluate(state: AgentState):
    """Montague Translation node."""
    print("🤖 [Jules] Decomposing request...")
    dag = parse_prompt(state["user_prompt"])
    
    # Initialize the primary Discourse box
    drs = DRS(label="main")
    
    return {
        "dag": dag.model_dump(),
        "drs": drs.to_dict(),
        "messages": [{"role": "assistant", "content": f"Structured logic into {len(dag.tasks)} frames."}]
    }

def execute_tasks(state: AgentState):
    """Honest Neurosymbolic Execution: MLTT checks + Frege feedback loop."""
    from orchestrator import tool_sandbox # Sandbox with 'forward' support
    
    # Restore the Discourse mental model
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger = CompositionLedger()
    
    tasks = state.get("dag", {}).get("tasks", [])
    overall_content = "### Formal Execution Phase\n\n"

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id")
        desc = task.get("description", "")
        
        # 1. DRT Anaphora Resolution: Augment description with accessible referents
        augmented_desc = drs.augment_description(desc)
        print(f"   ⏳ Task {i}: {augmented_desc[:100]}...")

        # 2. State Capture (Frege Start)
        before_state = WorkspaceState.capture(tool_sandbox)

        # 3. Execution (Montague -> Reality)
        prompt = f"Implement this task in the sandbox. Discourse Context:\n{augmented_desc}"
        res = chat_completion([{"role": "user", "content": prompt}])
        exec_out = execute_code(res)
        
        # 4. State Capture (Frege End)
        after_state = WorkspaceState.capture(tool_sandbox)
        
        # 5. Theoretical Update Loop
        delta = after_state.diff(before_state)
        ledger.record(task_id, before_state, after_state)
        drs.introduce_from_delta(task_id, delta) # Frege informs Discourse
        
        overall_content += f"#### Task {i}\n**Result:**\n{exec_out}\n\n"

    return {
        "drs": drs.to_dict(),
        "messages": [{"role": "assistant", "content": overall_content}]
    }

# --- Graph Wiring ---
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_tasks", execute_tasks)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_tasks")
workflow.add_edge("execute_tasks", END)
agent_executor = workflow.compile()
