import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END

from orchestrator.tool_sandbox import execute_code
from orchestrator.difficulty import compression_ratio, difficulty_to_tier
from orchestrator.supervisor import supervisor_node, conversation_node, direct_execute_node
from orchestrator.task_parser import parse_prompt
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.llm_client import chat_completion

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
    
    # 1. Parse the prompt into a structured TaskDAG using the LLM
    dag = parse_prompt(state["user_prompt"])
    
    # 2. Evaluate difficulty PER TASK instead of the original prompt length
    max_score = 0.0
    print(f"🧩 [Task Parser] Created {len(dag.tasks)} tasks:")
    for task in dag.tasks:
        # Score the individual task description
        score = compression_ratio(task.description)
        max_score = max(max_score, score)
        tier_string = difficulty_to_tier(score)
        print(f"   ↳ Task [{task.id[:4]}] Type: {task.task_type.upper():<10} | Tier: '{tier_string}' (Score: {score:.2f})")
        
    overall_tier = difficulty_to_tier(max_score)
    print(f"🧭 [Router] Overall DAG Execution Tier: '{overall_tier}'")
    
    # Format a response for Chainlit UI
    response_text = f"Decomposed request into {len(dag.tasks)} tasks. Maximum required execution tier: '{overall_tier}'.\n\n"
    for i, t in enumerate(dag.tasks, 1):
        response_text += f"{i}. **{t.task_type.upper()}**: {t.description}\n"
        
    return {
        "dag": dag.model_dump(),
        "messages": [{"role": "assistant", "content": response_text}]
    }

def execute_tasks(state: AgentState):
    """Executes the decomposed tasks sequentially."""
    print("🚀 [Jules] Executing task graph...")
    dag_dict = state.get("dag", {})
    tasks = dag_dict.get("tasks", [])
    
    execution_results = []
    overall_content = "### Task Execution Phase\n\n"

    for i, task in enumerate(tasks, 1):
        desc = task.get("description", "")
        tool = task.get("tool_hint", "python")
        task_type = task.get("task_type", "transform")
        
        print(f"   ⏳ Running Task {i}/{len(tasks)}: {desc[:50]}...")
        overall_content += f"#### Task {i}: {task_type.upper()}\n**Objective:** {desc}\n\n"
        
        # 1. Ask the LLM to generate the code for this specific atomic task
        sys_prompt = "You are an expert software agent. Write a Python script to accomplish the exact task provided. Output ONLY valid Python code wrapped in ```python ... ``` markdown blocks. Do not add conversational filler."
        user_req = f"Task Context: {state.get('user_prompt')}\n\nAtomic Task to implement: {desc}\n\nReturn the python code to execute this."
        
        raw_code_response = chat_completion(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_req}
            ],
            model="llm",
            temperature=0.1
        )
        
        # Determine the string output
        if hasattr(raw_code_response, 'choices'):
            code_output = raw_code_response.choices[0].message.content
        else:
            code_output = str(raw_code_response)

        overall_content += f"**Generated Code:**\n{code_output}\n\n"

        # 2. Execute the code in the sandbox
        if tool == "python" or not tool:  # default to python if empty
            exec_result = execute_code(code_output)
            overall_content += f"**Console Output:**\n```text\n{exec_result}\n```\n\n---\n\n"
        else:
            exec_result = f"Skipped: Tool '{tool}' is not currently supported in this sandbox."
            overall_content += f"**Console Output:**\n{exec_result}\n\n---\n\n"

        execution_results.append({
            "task_id": task.get("id"),
            "result": exec_result
        })

    return {
        "messages": [{"role": "assistant", "content": overall_content}],
        "final_output": {"results": execution_results}
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
