import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END

from orchestrator.tool_sandbox import execute_code
from orchestrator.difficulty import compression_ratio, difficulty_to_tier
from orchestrator.supervisor import supervisor_node, conversation_node, direct_execute_node
from orchestrator.task_parser import parse_prompt
from orchestrator.graph_memory import TriGraphMemory

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    next_node: str
    iteration: int
    user_prompt: str
    intent: str
    dag: dict
    final_output: dict

def decompose_and_evaluate(state: AgentState):
    print(f"🤖 [Jules] Decomposing prompt into tasks (Iteration {state.get('iteration', 0)})...")
    
    dag = parse_prompt(state["user_prompt"])
    
    max_score = 0.0
    # UPDATE THIS LINE: Change .tasks to whatever your schema uses (e.g., .nodes)
    print(f"🧩 [Task Parser] Created {len(dag.nodes)} tasks:") 
    
    # UPDATE THIS LINE TOO:
    for task in dag.nodes: 
        score = compression_ratio(task.description)
        max_score = max(max_score, score)
        tier_string = difficulty_to_tier(score)
        print(f"   ↳ Task [{task.id[:4]}] Type: {task.task_type.upper():<10} | Tier: '{tier_string}' (Score: {score:.2f})")
        
    overall_tier = difficulty_to_tier(max_score)
    print(f"🧭 [Router] Overall DAG Execution Tier: '{overall_tier}'")
    
    # Format a response for Chainlit UI
    response_text = f"Decomposed request into {len(dag.nodes)} tasks. Maximum required execution tier: '{overall_tier}'.\n\n"
    
    # UPDATE THIS LINE TOO:
    for i, t in enumerate(dag.nodes, 1): 
        response_text += f"{i}. **{t.task_type.upper()}**: {t.description}\n"
        
    return {
        "dag": dag.model_dump(),
        "messages": [{"role": "assistant", "content": response_text}]
    }

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

workflow.add_edge("conversation", END)
workflow.add_edge("direct_execute", END)
workflow.add_edge("decompose", END) 

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
