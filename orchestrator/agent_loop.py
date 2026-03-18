import operator
import os
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from orchestrator.tool_sandbox import execute_code
from orchestrator.task_parser import parse_prompt
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.llm_client import chat_completion

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict

def get_drt_universe(workspace_dir="/workspace") -> set:
    """DRT Context: Returns the set of all existing file paths in the sandbox discourse."""
    if not os.path.exists(workspace_dir): return set()
    files = set()
    for root, _, filenames in os.walk(workspace_dir):
        for f in filenames: files.add(os.path.join(root, f))
    return files

def decompose_and_evaluate(state: AgentState):
    """Montague/Fillmore Decomposition node."""
    dag = parse_prompt(state["user_prompt"])
    return {"dag": dag.model_dump(), "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} formal frames."}]}

def execute_tasks(state: AgentState):
    """Executes tasks formally with MLTT Type Checks and Frege Compositionality."""
    from orchestrator import tool_sandbox # Import for state capture
    ledger = CompositionLedger()
    print("🚀 [Jules] Executing task graph formally...")
    
    tasks = state.get("dag", {}).get("tasks", [])
    overall_content = "### Formal Execution Phase\n\n"
    
    for i, task in enumerate(tasks, 1):
        # 1. MLTT Type Check (Pre-execution Verification)
        drt_universe = get_drt_universe()
        target = task.get("target_resource", "")
        if task.get("task_type") in ("query", "verify") and target:
            if f"/workspace/{target}" not in drt_universe:
                overall_content += f"🛑 **MLTT Proof Failed**: {target} does not exist in discourse universe.\n\n"
                continue
        
        # 2. State Capture (Frege Start)
        before_state = WorkspaceState.capture(tool_sandbox)

        # 3. Code Generation and Execution
        res = chat_completion([{"role": "user", "content": f"Implement: {task['description']}"}])
        exec_out = execute_code(res)
        
        # 4. State Capture (Frege End)
        after_state = WorkspaceState.capture(tool_sandbox)
        ledger.record(task["id"], before_state, after_state)
        overall_content += f"Task {i} complete. Side-effects observed.\n\n"

    # Derive the final observed dependency graph based on Frege's Principle
    observed_graph = ledger.derive_dependency_graph()
    return {"messages": [{"role": "assistant", "content": overall_content}]}

# [Workflow construction and streaming logic remains as standard LangGraph boilerplate]
def route_intent(state: AgentState):
    # Read the classification from the supervisor node
    intent = state.get("intent", "DECOMPOSE")
    if intent == "CONVERSATION":
        return "conversation"
    elif intent == "DIRECT_EXECUTE":
        return "direct_execute"
    else:
        return "decompose"

# --- Build the LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("conversation", conversation_node)
workflow.add_node("direct_execute", direct_execute_node)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_tasks", execute_tasks)  # ADDED EXECUTION NODE

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor", 
    route_intent, 
    {
        "conversation": "conversation",
        "direct_execute": "direct_execute",
        "decompose": "decompose"
    }
)

# CHANGED: Graph structure now routes decompose -> execute_tasks -> END
workflow.add_edge("conversation", END)
workflow.add_edge("direct_execute", END)
workflow.add_edge("decompose", "execute_tasks") 
workflow.add_edge("execute_tasks", END)

agent_executor = workflow.compile()

async def agent_loop_stream(prompt: str):
    """Async generator for Chainlit to stream graph events."""
    initial_state = {
        "messages": [{"role": "user", "content": prompt}], 
        "user_prompt": prompt,
        "iteration": 0
    }
    
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            # Load the current memory state for the UI renderer
            mem = TriGraphMemory.load()
            
            # Yield the dictionary structure ui/web.py expects
            yield {
                "step": node,
                "state_snapshot": values,
                "node_update": values,
                "graphs": mem.to_dict()
            }
