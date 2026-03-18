import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END

from orchestrator.tool_sandbox import execute_code
from orchestrator.task_parser import parse_prompt
from orchestrator.llm_client import chat_completion
from orchestrator.graph_memory import TriGraphMemory

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    iteration: int

def decompose_and_evaluate(state: AgentState):
    print("🤖 Decomposing request...")
    dag = parse_prompt(state["user_prompt"])
    return {
        "dag": dag.model_dump(),
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} formal frames."}]
    }

def execute_tasks(state: AgentState):
    tasks = state.get("dag", {}).get("tasks", [])
    overall_content = "### Execution Phase\n\n"

    for i, task in enumerate(tasks, 1):
        desc = task.get("description", "")
        print(f"   ⏳ Task {i}: {desc[:80]}...")

        # Ask the Modal Inference pool to write the code
        prompt = f"Implement this task in Python/Bash: {desc}"
        res = chat_completion([{"role": "user", "content": prompt}])
        
        # Unpack the RPC response safely
        code_text = res.choices[0].message.content if hasattr(res, 'choices') else str(res)
            
        # Execute locally in the orchestrator container via subprocess
        exec_out = execute_code(code_text)
        
        overall_content += f"#### Task {i}\n**Result:**\n{exec_out}\n\n"

    return {"messages": [{"role": "assistant", "content": overall_content}]}

workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_tasks", execute_tasks)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_tasks")
workflow.add_edge("execute_tasks", END)
agent_executor = workflow.compile()

async def agent_loop_stream(prompt: str):
    """Provides the async stream expected by Chainlit in core_app.py"""
    initial_state = {"messages": [{"role": "user", "content": prompt}], "user_prompt": prompt, "iteration": 0}
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {"step": node, "state_snapshot": values, "node_update": values, "graphs": mem.to_dict()}
