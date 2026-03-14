import modal
from fleet_app import app

# Create a lightweight CPU image for the web UI. We explicitly add the fleet_app, orchestrator, inference, and ui
# modules to ensure we can import and invoke the orchestrator function AND that `chainlit run ui/web.py`
# can locate the file inside the container.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.135.1", "uvicorn>=0.41.0", "jinja2>=3.1.6", "python-multipart>=0.0.22",
        "pydantic>=2.12.5", "networkx>=3.6.1",
        "chainlit>=2.10.0",
        "llama-index-core>=0.14.17", "llama-index-embeddings-huggingface>=0.7.0",
        "smolagents>=1.24.0", "orjson>=3.11.7",
    )
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .add_local_python_source("inference", copy=True)
    .add_local_python_source("ui", copy=True)
    .add_local_file("fix_chainlit.py", "/tmp/fix_chainlit.py", copy=True)
    .run_commands(["python3 /tmp/fix_chainlit.py"])
)

try:
    import chainlit as cl
    import orjson

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

        # Track the latest Mermaid graph so we can show it after the loop ends
        final_graph_markdown = ""

        # Iterate over the generator from Modal
        _gen = run_agent_stream_func.remote_gen.aio(prompt)
        try:
            async for update in _gen:
                step_name = update["step"]
                state_snapshot = update["state_snapshot"]
                graphs_dict = update["graphs"]

                if step_name == "keep-alive":
                    # Just ignore keep-alive to keep connection open
                    continue

                # Display the execution step
                async with cl.Step(name=step_name) as step:
                    # Parse out a clean markdown display instead of raw JSON blocks
                    content_lines = []

                    if step_name == "Supervisor" and state_snapshot.get("intent"):
                        content_lines.append(f"**Intent Classified:** {state_snapshot['intent']}")

                    if step_name == "Retrieve_Codebase" and state_snapshot.get("codebase_context"):
                        ctx = state_snapshot["codebase_context"]
                        content_lines.append(f"**Codebase Context:**\n```python\n{ctx[:500]}...\n```")

                    # Show the task DAG cleanly if it's the Decompose step
                    if step_name == "Decompose" and "dag" in state_snapshot and state_snapshot["dag"]:
                        content_lines.append("**Tasks Decomposed:**")
                        tasks = state_snapshot["dag"].get("tasks", []) if isinstance(state_snapshot["dag"], dict) else getattr(state_snapshot["dag"], "tasks", [])
                        for t in tasks:
                            desc = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
                            content_lines.append(f"- {desc}")

                    # Show the latest messages
                    if "messages" in state_snapshot and state_snapshot["messages"]:
                        content_lines.append(f"**Agent:** {state_snapshot['messages'][-1]}")

                    # Show sandbox results if any
                    if "sandbox_results" in state_snapshot and state_snapshot["sandbox_results"]:
                        latest_sandbox = state_snapshot["sandbox_results"][-1]
                        # Handle both dict (JSON-serialised) and dataclass object (in-process)
                        if isinstance(latest_sandbox, dict):
                            stdout = latest_sandbox.get("stdout", "")
                            exit_code = latest_sandbox.get("exit_code", 0)
                        else:
                            stdout = getattr(latest_sandbox, "stdout", "")
                            exit_code = getattr(latest_sandbox, "exit_code", 0)
                        status_icon = "✅ Success" if exit_code == 0 else "❌ Failed"
                        content_lines.append(f"\n**Tool Execution ({status_icon}):**\n```\n{stdout[:1000]}\n```")

                    step.output = "\n".join(content_lines) if content_lines else orjson.dumps(state_snapshot, option=orjson.OPT_INDENT_2).decode("utf-8")

                # Render the episodic Tri-Graph as a native Mermaid diagram inside Chainlit
                mem = TriGraphMemory()
                mem.episodic = nx.node_link_graph(graphs_dict["episodic"])

                mermaid_lines = ["graph TD"]
                for node, data in mem.episodic.nodes(data=True):
                    label = str(data.get("description", node)).replace('"', "'")
                    if len(label) > 30:
                        label = label[:27] + "..."
                    status = data.get("status", "pending")
                    mermaid_lines.append(f'    {node}["{label}"]')
                    if status == "success":
                        mermaid_lines.append(f"    style {node} fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff")
                    elif status == "failed":
                        mermaid_lines.append(f"    style {node} fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff")
                    elif status == "running":
                        mermaid_lines.append(f"    style {node} fill:#f39c12,stroke:#d35400,stroke-width:2px,color:#fff")
                for u, v, _ in mem.episodic.edges(data=True):
                    mermaid_lines.append(f"    {u} --> {v}")

                graph_markdown = "\n".join(mermaid_lines)
                final_graph_markdown = graph_markdown  # persist latest state for finalize

                # Render Mermaid diagram as a fenced code block (Chainlit renders it natively)
                graph_msg.content = f"*Executing Node: {step_name}...*\n\n```mermaid\n{graph_markdown}\n```"
                graph_msg.elements = []
                await graph_msg.update()

            # Finalize — preserve the last rendered graph
            graph_msg.content = f"**Execution Complete. Final Graph State:**\n\n```mermaid\n{final_graph_markdown}\n```" if final_graph_markdown else "**Execution Complete.**"
            graph_msg.elements = []
            await graph_msg.update()
            await cl.Message(content="Task completed successfully.").send()

        except Exception as e:
            # Client disconnected or stream interrupted — update the graph message with
            # whatever was rendered so far rather than leaving the UI in a loading state.
            graph_msg.content = f"**Stream ended: {type(e).__name__}**"
            if final_graph_markdown:
                graph_msg.content += f"\n\n```mermaid\n{final_graph_markdown}\n```"
            graph_msg.elements = []
            await graph_msg.update()
        finally:
            # Always close the Modal generator so GeneratorExit is not silently ignored
            await _gen.aclose()

except ImportError:
    # This prevents CI pipelines and local modal deploys from failing when chainlit is not installed globally.
    # The code will still correctly execute when `chainlit run` is called from inside the Modal container where it is installed.
    pass


graph_state_vol = modal.Volume.from_name("dev_fleet-graph-state", create_if_missing=True)
workspace_vol = modal.Volume.from_name("dev_fleet-workspace", create_if_missing=True)


@app.function(
    image=web_image,
    volumes={
        "/state": graph_state_vol,
        "/workspace": workspace_vol,
    },
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
