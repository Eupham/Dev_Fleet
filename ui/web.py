import modal
from fleet_app import app

# Create a lightweight CPU image for the web UI. We explicitly add the fleet_app, orchestrator, inference, and ui
# modules to ensure we can import and invoke the orchestrator function AND that `chainlit run ui/web.py`
# can locate the file inside the container.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi", "uvicorn", "jinja2", "python-multipart", "pydantic>=2.5", "networkx>=3.2", "chainlit>=1.1.0", "pyvis>=0.3.2",
        "llama-index-core>=0.10.0", "llama-index>=0.10.0", "llama-index-embeddings-huggingface>=0.1.0"
    )
    .add_local_python_source("fleet_app")
    .add_local_python_source("orchestrator")
    .add_local_python_source("inference")
    .add_local_python_source("ui")
    .add_local_file("fix_chainlit.py", "/tmp/fix_chainlit.py")
    .run_commands(["python3 /tmp/fix_chainlit.py"])
)

try:
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

        # Need to use modal.Function.from_name inside the chainlit subprocess
        # since it runs separately from the main app's python context
        run_agent_stream_func = modal.Function.from_name("dev_fleet", "run_agent_stream")

        # Load elements conditionally
        loading_elements = [
            cl.Text(name="Status", content="Attempting to start Dev Fleet orchestration engine. Please wait. First boot might take up to 2 minutes.", display="inline")
        ]
        graph_msg.elements = loading_elements
        await graph_msg.update()

        # Iterate over the generator from Modal
        async for update in run_agent_stream_func.remote_gen.aio(prompt):
            step_name = update["step"]
            state_snapshot = update["state_snapshot"]
            graphs_dict = update["graphs"]

            # Display the execution step
            async with cl.Step(name=step_name) as step:
                # Parse out a clean markdown display instead of raw JSON blocks
                content_lines = []

                # Show the task DAG cleanly if it's the Decompose step
                if step_name == "Decompose" and "task_dag" in state_snapshot and state_snapshot["task_dag"]:
                    content_lines.append("**Tasks Decomposed:**")
                    tasks = state_snapshot["task_dag"].get("tasks", []) if isinstance(state_snapshot["task_dag"], dict) else getattr(state_snapshot["task_dag"], "tasks", [])
                    for t in tasks:
                        desc = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
                        content_lines.append(f"- {desc}")

                # Show the latest messages
                if "messages" in state_snapshot and state_snapshot["messages"]:
                    content_lines.append(f"**Agent:** {state_snapshot['messages'][-1]}")

                # Show sandbox results if any
                if "sandbox_results" in state_snapshot and state_snapshot["sandbox_results"]:
                    latest_sandbox = state_snapshot["sandbox_results"][-1]
                    content_lines.append(f"\n**Tool Execution Result:**\n```\n{latest_sandbox}\n```")

                step.output = "\n".join(content_lines) if content_lines else json.dumps(state_snapshot, indent=2, default=str)

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

except ImportError:
    # This prevents CI pipelines and local modal deploys from failing when chainlit is not installed globally.
    # The code will still correctly execute when `chainlit run` is called from inside the Modal container where it is installed.
    pass


@app.function(
    image=web_image,
    min_containers=1,
    timeout=3600,  # allow 1-hour WebSocket sessions (Modal default is 300s)
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000)
def ui():
    import subprocess
    # Run the chainlit server via subprocess.
    # Modal web_server expects a process to bind to the specified port to natively support WebSockets.
    subprocess.Popen(["chainlit", "run", "ui/web.py", "--host", "0.0.0.0", "--port", "8000", "--headless"])
