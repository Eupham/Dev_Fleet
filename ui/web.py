import modal
from fleet_app import app

# Create a lightweight CPU image for the web UI. We explicitly add the fleet_app, orchestrator, and inference modules
# to ensure we can import and invoke the orchestrator function.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi", "uvicorn", "jinja2", "python-multipart", "pydantic>=2.5", "networkx>=3.2", "chainlit>=1.1.0", "pyvis>=0.3.2",
        "llama-index-core>=0.10.0", "llama-index>=0.10.0", "llama-index-embeddings-huggingface>=0.1.0"
    )
    .add_local_python_source("fleet_app")
    .add_local_python_source("orchestrator")
    .add_local_python_source("inference")
)

import chainlit as cl
import json

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to Dev Fleet Orchestrator. Please enter your prompt to execute the agent. The Tri-Graph state will update as nodes execute.").send()

@cl.on_message
async def main(message: cl.Message):
    from orchestrator.core_app import run_agent_stream
    from orchestrator.graph_memory import TriGraphMemory, generate_interactive_graph_html
    import networkx as nx

    prompt = message.content

    # We will use a main message to attach the dynamically updating graph
    graph_msg = await cl.Message(content="*Initializing Agent...*").send()

    # Iterate over the generator from Modal
    async for update in run_agent_stream.remote_gen.aio(prompt):
        step_name = update["step"]
        state_snapshot = update["state_snapshot"]
        graphs_dict = update["graphs"]

        # Display the execution step
        async with cl.Step(name=step_name) as step:
            # We can dump the node's specific modifications to state as output
            # To keep it readable we just dump the new messages or status
            # For deeper inspection, users can expand the step
            out_content = {}
            if "messages" in state_snapshot and state_snapshot["messages"]:
                out_content["latest_message"] = state_snapshot["messages"][-1]
            if "sandbox_results" in state_snapshot and state_snapshot["sandbox_results"]:
                out_content["latest_sandbox"] = state_snapshot["sandbox_results"][-1]

            step.output = json.dumps(out_content, indent=2, default=str) if out_content else json.dumps(state_snapshot, indent=2, default=str)

        # Dynamically regenerate the HTML graph and update the main message elements
        # Reconstruct Memory object from dictionary to use the generator
        mem = TriGraphMemory()
        mem.semantic = nx.node_link_graph(graphs_dict["semantic"])
        mem.procedural = nx.node_link_graph(graphs_dict["procedural"])
        mem.episodic = nx.node_link_graph(graphs_dict["episodic"])

        graph_html = generate_interactive_graph_html(mem)

        elements = [
            cl.Html(
                name="Tri-Graph (Live)",
                content=graph_html,
                display="inline",
            )
        ]

        graph_msg.content = f"*Executing Node: {step_name}...*"
        graph_msg.elements = elements
        await graph_msg.update()

    # Finalize
    graph_msg.content = "**Execution Complete. Final Graph State:**"
    await graph_msg.update()
    await cl.Message(content="Task completed successfully.").send()


@app.function(
    image=web_image,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000)
def ui():
    import subprocess
    # Run the chainlit server via subprocess.
    # Modal web_server expects a process to bind to the specified port.
    subprocess.Popen(["chainlit", "run", "ui/web.py", "--host", "0.0.0.0", "--port", "8000", "--headless"])
