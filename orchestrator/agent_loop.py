import operator
from typing import Annotated, List, TypedDict, Union
from langgraph.graph import StateGraph, END
from orchestrator.llm_client import query_llm
from orchestrator.tool_sandbox import execute_code

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    next_node: str
    iteration: int

def call_model(state: AgentState):
    print(f"🤖 [Jules] Thinking (Iteration {state['iteration']})...")
    # Query the 'moderate' tier (Qwen 3.5 35B)
    response = query_llm(state["messages"], tier="moderate")
    
    # Simple logic to determine if we need a tool or we are done
    if "python" in response and "```" in response:
        return {"messages": [{"role": "assistant", "content": response}], "next_node": "execute"}
    return {"messages": [{"role": "assistant", "content": response}], "next_node": "end"}

def execute_tool(state: AgentState):
    print("🛠️ [Jules] Executing code in sandbox...")
    last_msg = state["messages"][-1]["content"]
    # Extract and run code
    result = execute_code(last_msg)
    return {
        "messages": [{"role": "user", "content": f"Sandbox Output:\n{result}"}],
        "iteration": state["iteration"] + 1,
        "next_node": "agent"
    }

def router(state: AgentState):
    return state["next_node"]

# Define the LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("execute", execute_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"execute": "execute", "end": END})
workflow.add_edge("execute", "agent")

# Compile the graph
agent_executor = workflow.compile()

async def run_agent_v2(prompt: str):
    initial_state = {"messages": [{"role": "user", "content": prompt}], "iteration": 0, "next_node": "agent"}
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            yield values["messages"][-1]["content"]
