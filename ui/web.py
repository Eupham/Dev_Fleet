import modal
from fleet_app import app

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.135.1", "uvicorn>=0.41.0", "jinja2>=3.1.6", "python-multipart>=0.0.22",
        "pydantic>=2.12.5", "networkx>=3.6.1", "mcp>=1.26.0", "langgraph>=1.1.2",
        "chainlit==2.10.0",  # pinned to match the monkey-patch target
        "llama-index-core>=0.14.17", "llama-index-embeddings-huggingface>=0.7.0",
        "orjson>=3.11.7", "pathspec>=1.0.4",
    )
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .add_local_python_source("inference", copy=True)
    .add_local_python_source("ui", copy=True)
    .env({"CHAINLIT_USER_ENV": "DUMMY_ENV_TO_PREVENT_NULL_CRASH"})
    .add_local_dir(".chainlit", remote_path="/root/.chainlit", copy=True)
    .add_local_dir("public", remote_path="/root/public", copy=True)
)

try:
    import json
    import chainlit as cl
    import orjson

    # ---------------------------------------------------------------------------
    # Mermaid shape helpers — different shapes per node label type
    # ---------------------------------------------------------------------------

    # Maps label → (open_bracket, close_bracket) for Mermaid node syntax
    # nodeId[open_bracket "text" close_bracket]  (the brackets are literal chars)
    _LABEL_SHAPE: dict[str, tuple[str, str]] = {
        # Episodic / task
        "Task":       ("[",  "]"),   # rectangle
        # Semantic
        "Concept":    ("(",  ")"),   # rounded rectangle
        "Entity":     ("(",  ")"),
        "Fact":       ("[",  "]"),
        "Principle":  ("{",  "}"),   # rhombus
        "Pattern":    ("[/", "/]"),  # parallelogram
        "Definition": ("[",  "]"),
        # Procedural
        "Tool":       ("([", "])"),  # stadium
        "Function":   ("[/", "/]"),  # parallelogram
        "Step":       ("[",  "]"),
        "Rule":       ("{",  "}"),   # rhombus
        "Workflow":   ("([", "])"),  # stadium
        "Template":   ("[",  "]"),
    }
    _DEFAULT_SHAPE = ("[", "]")

    def _mermaid_node(node_id: str, label: str, text: str) -> str:
        """Return a single Mermaid node declaration line."""
        o, c = _LABEL_SHAPE.get(label, _DEFAULT_SHAPE)
        safe_text = text.replace('"', "#quot;").replace("'", "#39;")
        return f'        {node_id}{o}"{safe_text}"{c}'

    def _safe_label(text: str, max_len: int = 60) -> str:
        text = str(text)
        return text if len(text) <= max_len else text[:max_len - 3] + "..."

    def _nid(prefix: str, raw: str) -> str:
        return f"{prefix}_{str(raw).replace('-', '_').replace('.', '_')}"

    # ---------------------------------------------------------------------------
    # Task list renderer
    # ---------------------------------------------------------------------------

    _STATUS_ICON = {
        "success": "✅",
        "failed":  "❌",
        "running": "🔄",
        "pending": "⬜",
    }

    def _render_task_list(dag: dict | None, current_idx: int = 0) -> str:
        if not dag:
            return "*No tasks yet.*"
        tasks = dag.get("tasks", []) if isinstance(dag, dict) else getattr(dag, "tasks", [])
        if not tasks:
            return "*No tasks yet.*"
        lines = ["**Task List**\n"]
        for i, t in enumerate(tasks):
            desc = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
            status = t.get("status", "pending") if isinstance(t, dict) else getattr(t, "status", "pending")
            hint = t.get("tool_hint", "") if isinstance(t, dict) else getattr(t, "tool_hint", "")
            icon = _STATUS_ICON.get(status, "⬜")
            hint_tag = f" `[{hint}]`" if hint else ""
            lines.append(f"{i + 1}. {icon} {desc}{hint_tag}")
        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Tri-Graph renderer — Mermaid markup wrapped in mermaid.js HTML
    # ---------------------------------------------------------------------------


    def _render_trigraph(graphs_dict: dict, nx) -> str:
        """Return HTML string with a mermaid diagram (rendered via CDN script)."""
        from orchestrator.graph_memory import TriGraphMemory

        mem = TriGraphMemory()
        mem.episodic   = nx.node_link_graph(graphs_dict["episodic"])
        mem.semantic   = nx.node_link_graph(graphs_dict["semantic"])
        mem.procedural = nx.node_link_graph(graphs_dict["procedural"])

        has_ep   = mem.episodic.number_of_nodes() > 0
        has_sem  = mem.semantic.number_of_nodes() > 0
        has_proc = mem.procedural.number_of_nodes() > 0

        if not (has_ep or has_sem or has_proc):
            return ""

        # Use compatible subgraph syntax: subgraph id [title] (no quoted id)
        lines = ["graph TD"]

        if has_ep:
            lines.append("    subgraph ep_group [Task Execution]")
            for node, data in mem.episodic.nodes(data=True):
                nid   = _nid("ep", node)
                label = data.get("label", "Task")
                text  = _safe_label(data.get("content") or data.get("description") or str(node))
                lines.append(_mermaid_node(nid, label, text))
                status = data.get("status", "pending")
                if status == "success":
                    lines.append(f"        style {nid} fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff")
                elif status == "failed":
                    lines.append(f"        style {nid} fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff")
                elif status == "running":
                    lines.append(f"        style {nid} fill:#f39c12,stroke:#d35400,stroke-width:2px,color:#fff")
                else:
                    lines.append(f"        style {nid} fill:#95a5a6,stroke:#7f8c8d,stroke-width:1px,color:#fff")
            lines.append("    end")

        if has_sem:
            lines.append("    subgraph sem_group [Semantic Knowledge]")
            for node, data in mem.semantic.nodes(data=True):
                nid   = _nid("sem", node)
                label = data.get("label", "Concept")
                text  = _safe_label(data.get("content") or data.get("description") or data.get("name") or str(node))
                lines.append(_mermaid_node(nid, label, text))
                lines.append(f"        style {nid} fill:#3498db,stroke:#2980b9,stroke-width:1px,color:#fff")
            lines.append("    end")

        if has_proc:
            lines.append("    subgraph proc_group [Procedural Rules]")
            for node, data in mem.procedural.nodes(data=True):
                nid   = _nid("proc", node)
                label = data.get("label", "Tool")
                text  = _safe_label(data.get("content") or data.get("description") or data.get("rule") or str(node))
                lines.append(_mermaid_node(nid, label, text))
                lines.append(f"        style {nid} fill:#27ae60,stroke:#229954,stroke-width:1px,color:#fff")
            lines.append("    end")

        for u, v in mem.episodic.edges():
            lines.append(f"    {_nid('ep', u)} --> {_nid('ep', v)}")
        for u, v in mem.semantic.edges():
            lines.append(f"    {_nid('sem', u)} --> {_nid('sem', v)}")
        for u, v in mem.procedural.edges():
            lines.append(f"    {_nid('proc', u)} --> {_nid('proc', v)}")

        mermaid_src = "\n".join(lines)
        return f"```mermaid\n{mermaid_src}\n```"

    # ---------------------------------------------------------------------------
    # Knowledge Graph Browser — persistent action button + callback
    # ---------------------------------------------------------------------------

    @cl.on_chat_start
    async def on_chat_start():
        """Send a persistent action button so the user can inspect the knowledge
        graph at any time — including after a session has finished."""
        await cl.Message(
            content=(
                "**Dev Fleet** is ready.\n\n"
                "Use the button below to browse the persisted knowledge graph "
                "at any time, even after a run completes."
            ),
            actions=[
                cl.Action(
                    name="browse_knowledge",
                    label="Browse Knowledge Graph",
                    description="Load and display the full Tri-Graph knowledge state from persistent storage",
                    payload={},
                )
            ],
        ).send()

    @cl.action_callback("browse_knowledge")
    async def on_browse_knowledge(action: cl.Action):
        """Load the persisted knowledge graphs and render them as a Mermaid diagram."""
        import networkx as nx

        read_graph_state_func = modal.Function.from_name("dev_fleet", "read_graph_state")

        async with cl.Step(name="Knowledge Graph Browser") as step:
            step.output = "Loading persisted graphs from storage..."

        try:
            graph_data = await read_graph_state_func.remote.aio()
        except Exception as e:
            await cl.Message(
                content=f"**Knowledge Graph Browser — Error**\n\n`{type(e).__name__}: {e}`"
            ).send()
            return

        if not graph_data.get("persisted"):
            await cl.Message(
                content=(
                    "**Knowledge Graph Browser**\n\n"
                    "*No persisted graph state found. Run a task first to populate the graphs.*"
                )
            ).send()
            return

        total  = graph_data.get("total_nodes", 0)
        ep_n   = graph_data["episodic"]["count"]
        sem_n  = graph_data["semantic"]["count"]
        proc_n = graph_data["procedural"]["count"]

        # Build NetworkX graphs from the flat node/edge lists
        def _build_nx(graph_dict: dict) -> nx.DiGraph:
            g = nx.DiGraph()
            for node in graph_dict.get("nodes", []):
                g.add_node(
                    node["id"],
                    label=node.get("label", ""),
                    content=node.get("content", ""),
                    status=node.get("status", ""),
                )
            for edge in graph_dict.get("edges", []):
                g.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))
            return g

        # Reconstruct a graphs_dict compatible with _render_trigraph
        graphs_dict_for_render = {
            "episodic":   nx.node_link_data(_build_nx(graph_data["episodic"]),   edges="edges"),
            "semantic":   nx.node_link_data(_build_nx(graph_data["semantic"]),   edges="edges"),
            "procedural": nx.node_link_data(_build_nx(graph_data["procedural"]), edges="edges"),
        }

        mermaid_str = _render_trigraph(graphs_dict_for_render, nx)
        summary = (
            f"**Tri-Graph Knowledge Browser**\n\n"
            f"**Total nodes:** {total} "
            f"(Episodic: {ep_n} | Semantic: {sem_n} | Procedural: {proc_n})\n\n"
        )

        await cl.Message(
            content=summary + (mermaid_str if mermaid_str else "*All graphs are empty.*"),
            actions=[
                cl.Action(
                    name="browse_knowledge",
                    label="Refresh Knowledge Graph",
                    description="Reload the latest persisted graph state",
                    payload={},
                )
            ],
        ).send()

    # ---------------------------------------------------------------------------
    # Message handler
    # ---------------------------------------------------------------------------

    @cl.on_message
    async def main(message: cl.Message):
        import networkx as nx
        from orchestrator.core_app import run_agent_stream

        prompt = message.content
        run_agent_stream_func = modal.Function.from_name("dev_fleet", "run_agent_stream")

        # Three separate persistent messages:
        #   task_msg      — the task list (lazy; created once DAG is available)
        #   workspace_msg — live workspace diary (lazy; created after first Execute)
        #   graph_msg     — the Tri-Graph knowledge visualization
        task_msg:      cl.Message | None = None
        workspace_msg: cl.Message | None = None
        graph_msg: cl.Message = await cl.Message(
            content="**Tri-Graph Knowledge State**\n\n*Waiting for first node...*"
        ).send()

        final_graph_html = ""

        _gen = run_agent_stream_func.remote_gen.aio(prompt)
        try:
            async for update in _gen:
                step_name      = update["step"]
                state_snapshot = update["state_snapshot"]
                graphs_dict    = update["graphs"]
                node_update    = update.get("node_update", {})
                # model_info and sandbox_results are hoisted to the top-level
                # yield dict by agent_loop_stream so they are always available.
                step_model_info      = update.get("model_info") or node_update.get("model_info") or {}
                step_sandbox_results = update.get("sandbox_results") or node_update.get("sandbox_results") or []
                # Execution context
                step_task_desc   = update.get("task_desc") or node_update.get("task_desc") or ""
                step_task_idx    = update.get("task_idx")  or node_update.get("task_idx")  or 0
                step_total_tasks = update.get("total_tasks") or node_update.get("total_tasks") or 0
                step_subtask_ids = update.get("subtask_ids") or node_update.get("subtask_ids") or []
                step_gpu_uptime  = update.get("gpu_uptime_s") or node_update.get("gpu_uptime_s") or 0.0

                if step_name == "keep-alive":
                    continue

                # Use the task description as the step name for execute steps
                # so the sidebar shows "Task 2/5: Conduct research…" instead of
                # the node name "execute_single_task".
                if step_name == "execute_single_task" and step_task_desc:
                    display_name = (
                        f"Task {step_task_idx}/{step_total_tasks}: "
                        f"{step_task_desc[:60]}{'...' if len(step_task_desc) > 60 else ''}"
                    )
                else:
                    display_name = step_name

                async with cl.Step(name=display_name) as step:
                    content_lines = []

                    if step_name == "Supervisor":
                        intent = node_update.get("intent") or state_snapshot.get("intent", "unknown")
                        intent_desc = {
                            "DECOMPOSE":      "multi-step task — routing to full decomposition pipeline",
                            "DIRECT_EXECUTE": "single-step task — bypassing decomposition",
                            "CONVERSATION":   "conversational query — routing to direct response",
                            "RESEARCH":       "web research — fetching online data before decomposition",
                        }.get(intent, intent)
                        content_lines.append(f"**Classified:** {intent} — {intent_desc}")

                    elif step_name == "Retrieve_Codebase":
                        ctx = node_update.get("codebase_context") or state_snapshot.get("codebase_context", "")
                        if ctx:
                            content_lines.append(f"**Relevant codebase context:**\n```\n{ctx}\n```")
                        else:
                            content_lines.append("**Codebase scan:** No relevant files found — proceeding with empty context.")

                    elif step_name in ("Decompose", "decompose"):
                        dag = node_update.get("dag") or state_snapshot.get("dag")
                        if dag:
                            tasks = dag.get("tasks", []) if isinstance(dag, dict) else getattr(dag, "tasks", [])
                            diff_scores = (
                                node_update.get("difficulty_scores")
                                or state_snapshot.get("difficulty_scores")
                                or {}
                            )
                            content_lines.append(f"**{len(tasks)} tasks decomposed:**")
                            for i, t in enumerate(tasks, 1):
                                desc  = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
                                hint  = t.get("tool_hint", "")   if isinstance(t, dict) else getattr(t, "tool_hint", "")
                                tid   = t.get("id", "")           if isinstance(t, dict) else getattr(t, "id", "")
                                score_meta = diff_scores.get(tid, {})
                                tier_tag   = f" `{score_meta.get('tier', '')} K={score_meta.get('score', 0):.2f}`" if score_meta else ""
                                hint_tag   = f" `[{hint}]`" if hint else ""
                                content_lines.append(f"{i}. {desc}{hint_tag}{tier_tag}")

                    elif step_name == "Rerank_and_Retrieve":
                        msgs = node_update.get("messages", [])
                        dag  = state_snapshot.get("dag")
                        task_count = len(dag.get("tasks", [])) if dag and isinstance(dag, dict) else 0
                        if msgs:
                            content_lines.append(f"**{msgs[-1]}**")
                        if task_count:
                            content_lines.append(
                                f"Scored {task_count} task{'s' if task_count != 1 else ''} "
                                "against semantic and procedural graphs."
                            )

                    elif step_name == "execute_single_task":
                        # --- Model / GPU badge with uptime ---
                        if step_model_info:
                            tier  = step_model_info.get("tier", "?")
                            model = step_model_info.get("model", "?")
                            gpu   = step_model_info.get("gpu", "?")
                            uptime_str = (
                                f"{int(step_gpu_uptime // 60)}m {int(step_gpu_uptime % 60)}s"
                                if step_gpu_uptime >= 60
                                else f"{step_gpu_uptime:.1f}s"
                            )
                            content_lines.append(
                                f"**Model:** `{model}` | **GPU:** `{gpu}` "
                                f"| **Tier:** `{tier}` | **GPU uptime:** `{uptime_str}`"
                            )

                        # --- Spawned sub-tasks ---
                        if step_subtask_ids:
                            dag_tasks = (
                                state_snapshot.get("dag", {}).get("tasks", [])
                                or node_update.get("dag", {}).get("tasks", [])
                            )
                            spawned_descs = [
                                t.get("description", tid)
                                for t in dag_tasks
                                for tid in step_subtask_ids
                                if (t.get("id") == tid)
                            ]
                            if spawned_descs:
                                content_lines.append(
                                    f"\n**Spawned {len(spawned_descs)} sub-task(s):**"
                                )
                                for i, sd in enumerate(spawned_descs, 1):
                                    content_lines.append(f"  {i}. {sd}")

                        # --- Tool call results ---
                        if step_sandbox_results:
                            content_lines.append(
                                f"\n**Tool calls ({len(step_sandbox_results)} total):**"
                            )
                            for r in step_sandbox_results:
                                tool   = r.get("tool", "?") if isinstance(r, dict) else getattr(r, "tool", "?")
                                args   = r.get("args", {})   if isinstance(r, dict) else getattr(r, "args", {})
                                output = r.get("output", "") if isinstance(r, dict) else getattr(r, "output", "")

                                args_display = json.dumps(args)
                                if len(args_display) > 120:
                                    args_display = args_display[:120] + "..."

                                if tool == "[model_text]":
                                    content_lines.append(f"\n**[model reasoning]**\n```\n{output[:600]}\n```")
                                else:
                                    content_lines.append(
                                        f"\n**`{tool}`** `{args_display}`\n"
                                        f"```\n{output[:800]}\n```"
                                    )
                        else:
                            msgs = node_update.get("messages", [])
                            if msgs:
                                content_lines.append(msgs[-1])

                    elif step_name == "Handle_Failure":
                        attempt = state_snapshot.get("current_attempt", 1)
                        content_lines.append(f"**Retrying task** (attempt {attempt} of 2)...")
                        msgs = node_update.get("messages", [])
                        if msgs:
                            content_lines.append(f"\n{msgs[-1]}")

                    elif step_name in ("Conversation", "Direct_Execute"):
                        msgs = node_update.get("messages", [])
                        if msgs:
                            content_lines.append(msgs[-1])

                    step.output = "\n".join(content_lines) if content_lines else f"*{step_name} completed*"

                # --- Task list (separate persistent message, created lazily) ---
                current_dag = (
                    node_update.get("dag")
                    or state_snapshot.get("dag")
                )
                if current_dag:
                    task_list_md = _render_task_list(
                        current_dag,
                        state_snapshot.get("current_task_idx", 0),
                    )
                    if task_msg is None:
                        task_msg = await cl.Message(content=task_list_md).send()
                    else:
                        task_msg.content = task_list_md
                        await task_msg.update()

                # --- Workspace diary (separate persistent message, created lazily) ---
                ctx = (
                    node_update.get("codebase_context")
                    or state_snapshot.get("codebase_context", "")
                )
                if ctx:
                    workspace_md = f"**Workspace**\n\n```\n{ctx}\n```"
                    if workspace_msg is None:
                        workspace_msg = await cl.Message(content=workspace_md).send()
                    else:
                        workspace_msg.content = workspace_md
                        await workspace_msg.update()

                # --- Tri-Graph visualization (separate persistent message) ---
                try:
                    graph_html = _render_trigraph(graphs_dict, nx)
                    if graph_html:
                        final_graph_html = graph_html
                        graph_msg.content = f"**Tri-Graph Knowledge State**\n\n{graph_html}"
                    else:
                        graph_msg.content = "**Tri-Graph Knowledge State**\n\n*Graph is empty — no nodes yet.*"
                    graph_msg.elements = []
                    await graph_msg.update()
                except Exception as graph_exc:
                    import logging
                    logging.getLogger("dev_fleet.ui").warning("Graph render error: %s", graph_exc)
                    graph_msg.content = f"**Tri-Graph Knowledge State**\n\n*Render error: {type(graph_exc).__name__}: {graph_exc}*"
                    graph_msg.elements = []
                    await graph_msg.update()

            # Finalize
            if final_graph_html:
                graph_msg.content = f"**Tri-Graph Knowledge State — Final**\n\n{final_graph_html}"
                graph_msg.elements = []
                await graph_msg.update()

        except Exception as e:
            graph_msg.content = f"**Error: {type(e).__name__}: {e}**"
            if final_graph_html:
                graph_msg.content += f"\n\n{final_graph_html}"
            graph_msg.elements = []
            await graph_msg.update()
        finally:
            await _gen.aclose()

except ImportError:
    pass


graph_state_vol = modal.Volume.from_name("dev_fleet-graph-state", create_if_missing=True)
workspace_vol   = modal.Volume.from_name("dev_fleet-workspace",    create_if_missing=True)


@app.function(
    image=web_image,
    volumes={
        "/state":     graph_state_vol,
        "/workspace": workspace_vol,
    },
    min_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=60)
def ui():
    import subprocess
    subprocess.Popen(["chainlit", "run", "ui/web.py", "--host", "0.0.0.0", "--port", "8000", "--headless"])
