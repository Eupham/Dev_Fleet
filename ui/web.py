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
        "smolagents>=1.24.0", "orjson>=3.11.7", "pathspec>=0.12.1",
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

    @cl.on_message
    async def main(message: cl.Message):
        from orchestrator.core_app import run_agent_stream
        from orchestrator.graph_memory import TriGraphMemory
        import networkx as nx

        prompt = message.content

        run_agent_stream_func = modal.Function.from_name("dev_fleet", "run_agent_stream")

        # Single persistent graph message — updates live as nodes execute
        graph_msg = await cl.Message(content="**Tri-Graph Knowledge State**\n\n*Waiting for first node...*").send()

        final_graph_markdown = ""

        _gen = run_agent_stream_func.remote_gen.aio(prompt)
        try:
            async for update in _gen:
                step_name = update["step"]
                state_snapshot = update["state_snapshot"]
                graphs_dict = update["graphs"]
                node_update = update.get("node_update", {})

                if step_name == "keep-alive":
                    continue

                # --- Per-step panel with clean, meaningful content ---
                async with cl.Step(name=step_name) as step:
                    content_lines = []

                    if step_name == "Supervisor":
                        intent = node_update.get("intent") or state_snapshot.get("intent", "unknown")
                        intent_desc = {
                            "DECOMPOSE": "multi-step task — routing to full decomposition pipeline",
                            "DIRECT_EXECUTE": "single-step task — bypassing decomposition",
                            "CONVERSATION": "conversational query — routing to direct response",
                        }.get(intent, intent)
                        content_lines.append(f"**Classified:** {intent} — {intent_desc}")

                    elif step_name == "Retrieve_Codebase":
                        ctx = node_update.get("codebase_context") or state_snapshot.get("codebase_context", "")
                        if ctx:
                            content_lines.append(f"**Relevant codebase context:**\n```\n{ctx}\n```")
                        else:
                            content_lines.append("**Codebase scan:** No relevant files found — proceeding with empty context.")

                    elif step_name == "Decompose":
                        dag = node_update.get("dag") or state_snapshot.get("dag")
                        if dag:
                            tasks = dag.get("tasks", []) if isinstance(dag, dict) else getattr(dag, "tasks", [])
                            content_lines.append(f"**{len(tasks)} tasks decomposed:**")
                            for i, t in enumerate(tasks, 1):
                                desc = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
                                hint = t.get("tool_hint", "") if isinstance(t, dict) else getattr(t, "tool_hint", "")
                                hint_tag = f" `[{hint}]`" if hint else ""
                                deps = t.get("depends_on", []) if isinstance(t, dict) else getattr(t, "depends_on", [])
                                dep_tag = f" *(depends on {len(deps)} task{'s' if len(deps) != 1 else ''})*" if deps else ""
                                content_lines.append(f"{i}. {desc}{hint_tag}{dep_tag}")

                    elif step_name == "Rerank_and_Retrieve":
                        msgs = node_update.get("messages", [])
                        dag = state_snapshot.get("dag")
                        task_count = len(dag.get("tasks", [])) if dag and isinstance(dag, dict) else 0
                        if msgs:
                            content_lines.append(f"**{msgs[-1]}**")
                        if task_count:
                            content_lines.append(f"Scored {task_count} task{'s' if task_count != 1 else ''} against semantic and procedural graphs.")

                    elif step_name == "Execute":
                        results = node_update.get("sandbox_results", [])
                        if results:
                            latest = results[-1]
                            stdout = latest.get("stdout", "") if isinstance(latest, dict) else getattr(latest, "stdout", "")
                            stderr = latest.get("stderr", "") if isinstance(latest, dict) else getattr(latest, "stderr", "")
                            exit_code = latest.get("exit_code", 0) if isinstance(latest, dict) else getattr(latest, "exit_code", 0)
                            status = "✅ Success" if exit_code == 0 else "❌ Failed"
                            if stdout:
                                content_lines.append(f"**{status}:**\n```\n{stdout[:2000]}\n```")
                            elif exit_code == 0:
                                content_lines.append(f"**{status}** (no output)")
                            if stderr and exit_code != 0:
                                content_lines.append(f"**Stderr:**\n```\n{stderr[:500]}\n```")
                        else:
                            msgs = node_update.get("messages", [])
                            if msgs:
                                content_lines.append(msgs[-1])

                    elif step_name == "Handle_Failure":
                        attempt = state_snapshot.get("current_attempt", 1)
                        content_lines.append(f"**Retrying task** (attempt {attempt} of 2)...")

                    elif step_name in ("Conversation", "Direct_Execute"):
                        msgs = node_update.get("messages", [])
                        if msgs:
                            content_lines.append(msgs[-1])

                    step.output = "\n".join(content_lines) if content_lines else f"*{step_name} completed*"

                # --- Live Tri-Graph visualization ---
                try:
                    mem = TriGraphMemory()
                    mem.episodic = nx.node_link_graph(graphs_dict["episodic"])
                    mem.semantic = nx.node_link_graph(graphs_dict["semantic"])
                    mem.procedural = nx.node_link_graph(graphs_dict["procedural"])

                    def _safe_label(text: str, max_len: int = 60) -> str:
                        text = str(text).replace('"', "'")
                        return text if len(text) <= max_len else text[:max_len - 3] + "..."

                    def _nid(prefix: str, raw: str) -> str:
                        safe = str(raw).replace("-", "_").replace(".", "_")
                        return f"{prefix}_{safe}"

                    mermaid_lines = ["graph TD"]

                    has_episodic = mem.episodic.number_of_nodes() > 0
                    has_semantic = mem.semantic.number_of_nodes() > 0
                    has_procedural = mem.procedural.number_of_nodes() > 0

                    if has_episodic:
                        mermaid_lines.append('    subgraph EPISODIC["Task Execution"]')
                        for node, data in mem.episodic.nodes(data=True):
                            nid = _nid("ep", node)
                            label = _safe_label(data.get("description", node))
                            status = data.get("status", "pending")
                            mermaid_lines.append(f'        {nid}["{label}"]')
                            if status == "success":
                                mermaid_lines.append(f"        style {nid} fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff")
                            elif status == "failed":
                                mermaid_lines.append(f"        style {nid} fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff")
                            elif status == "running":
                                mermaid_lines.append(f"        style {nid} fill:#f39c12,stroke:#d35400,stroke-width:2px,color:#fff")
                            else:
                                mermaid_lines.append(f"        style {nid} fill:#95a5a6,stroke:#7f8c8d,stroke-width:1px,color:#fff")
                        mermaid_lines.append("    end")

                    if has_semantic:
                        mermaid_lines.append('    subgraph SEMANTIC["Semantic Knowledge"]')
                        for node, data in mem.semantic.nodes(data=True):
                            nid = _nid("sem", node)
                            label = _safe_label(data.get("description") or data.get("name") or str(node))
                            mermaid_lines.append(f'        {nid}["{label}"]')
                            mermaid_lines.append(f"        style {nid} fill:#3498db,stroke:#2980b9,stroke-width:1px,color:#fff")
                        mermaid_lines.append("    end")

                    if has_procedural:
                        mermaid_lines.append('    subgraph PROCEDURAL["Procedural Rules"]')
                        for node, data in mem.procedural.nodes(data=True):
                            nid = _nid("proc", node)
                            label = _safe_label(data.get("description") or data.get("rule") or str(node))
                            mermaid_lines.append(f'        {nid}["{label}"]')
                            mermaid_lines.append(f"        style {nid} fill:#27ae60,stroke:#229954,stroke-width:1px,color:#fff")
                        mermaid_lines.append("    end")

                    # Edges
                    for u, v in mem.episodic.edges():
                        mermaid_lines.append(f"    {_nid('ep', u)} --> {_nid('ep', v)}")
                    for u, v in mem.semantic.edges():
                        mermaid_lines.append(f"    {_nid('sem', u)} --> {_nid('sem', v)}")
                    for u, v in mem.procedural.edges():
                        mermaid_lines.append(f"    {_nid('proc', u)} --> {_nid('proc', v)}")

                    graph_markdown = "\n".join(mermaid_lines)
                    final_graph_markdown = graph_markdown

                    if has_episodic or has_semantic or has_procedural:
                        graph_msg.content = f"**Tri-Graph Knowledge State**\n\n```mermaid\n{graph_markdown}\n```"
                    else:
                        graph_msg.content = "**Tri-Graph Knowledge State**\n\n*Graph is empty — no nodes yet.*"
                    graph_msg.elements = []
                    await graph_msg.update()

                except Exception:
                    pass  # Never let graph rendering break the step loop

            # Finalize
            if final_graph_markdown:
                graph_msg.content = f"**Tri-Graph Knowledge State — Final**\n\n```mermaid\n{final_graph_markdown}\n```"
            else:
                graph_msg.content = "**Execution Complete**"
            graph_msg.elements = []
            await graph_msg.update()

        except Exception as e:
            graph_msg.content = f"**Error: {type(e).__name__}: {e}**"
            if final_graph_markdown:
                graph_msg.content += f"\n\n```mermaid\n{final_graph_markdown}\n```"
            graph_msg.elements = []
            await graph_msg.update()
        finally:
            await _gen.aclose()

except ImportError:
    # Prevents CI pipelines and local modal deploys from failing when chainlit is not installed globally.
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
