"""LangGraph agent loop — with tool-use execution, difficulty reassessment,
and sub-task spawning.

FIXES APPLIED (v2 — against actual repo):
1. EXECUTION WAS A NO-OP: `chat_completion` was called once, the raw response
   string was passed to `execute_code(str(res))` which regex-extracts code
   blocks. If the model didn't produce a markdown code block, nothing happened.
   Even if it did, it was ONE shot — no iteration, no tool use.

   Fixed: Iterative tool-use loop where the model gets tool definitions
   (`web_search`, `execute_code`, `write_file`, `read_file`, `task_complete`)
   and calls them in a loop until signaling completion or hitting max iterations.

   This requires the matching fixes in:
   - llm_client.py (forwards `tools`/`tool_choice` to Modal)
   - inference/utils.py (BaseInference.generate_logic returns full response
     object when tools are present)

2. LEDGER BUG: CompositionLedger deserialization via from_dict().
3. DIFFICULTY REASSESSMENT: After each task, remaining tasks re-scored.
4. SUB-TASK SPAWNING: Deterministic heuristics for error recovery.
5. MULTI-TIER ROUTING: difficulty tier passed to chat_completion.
6. WorkspaceState.capture() guarded against missing sandbox.
"""

import operator
import os
import json
import re
from typing import Annotated, List, TypedDict, Optional

from langgraph.graph import StateGraph, END
import networkx as nx

from orchestrator.tool_sandbox import execute_code, forward
from orchestrator.task_parser import parse_prompt, TaskDAG, TransformTask, VerifyTask
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS, ReferentType
from orchestrator.llm_client import chat_completion, ModalKeepAlive
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.difficulty import (
    compute_base_difficulty, difficulty_to_tier, reassess_remaining_tasks
)
from orchestrator.indexer import build_knowledge_graphs


# ---------------------------------------------------------------------------
# Tool Definitions (forwarded to llama-server via OpenAI-compatible API)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use for research tasks, finding documentation, discovering libraries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python or bash code in the sandbox. Returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "bash"]},
                    "code": {"type": "string", "description": "Code to execute"},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to /workspace"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to /workspace"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Signal that the current task is finished. MUST be called when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Brief summary of what was accomplished"},
                },
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Dispatch — uses actual tool_sandbox.forward()
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a tool call via the actual tool_sandbox module.

    Routes to tool_sandbox.forward() which uses subprocess to run
    code in the sandbox environment. This is the bridge between
    the LLM's tool calls and actual system execution.
    """
    try:
        if name == "web_search":
            query = arguments.get("query", "")
            # Run web search via Python in sandbox
            code = f"""
try:
    from duckduckgo_search import DDGS
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "duckduckgo-search"], capture_output=True)
    from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text({repr(query)}, max_results=5))
    for r in results:
        print(f"## {{r.get('title', '')}}")
        print(f"{{r.get('body', '')}}")
        print(f"URL: {{r.get('href', '')}}")
        print()
"""
            result = forward(code=code, language="python", timeout=30)
            return result.get("stdout", "") or "No search results."

        elif name == "run_code":
            language = arguments.get("language", "python")
            code = arguments.get("code", "")
            result = forward(code=code, language=language, timeout=30)
            return result.get("stdout", "") or "Code executed (no output)."

        elif name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            # Use json.dumps for safe string embedding
            code = f"""
import os, json
path = {json.dumps(path)}
content = {json.dumps(content)}
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as f:
    f.write(content)
print(f"Written {{len(content)}} chars to {{path}}")
"""
            result = forward(code=code, language="python", timeout=10)
            return result.get("stdout", "") or "File written."

        elif name == "read_file":
            path = arguments.get("path", "")
            code = f"""
path = {json.dumps(path)}
try:
    with open(path, "r") as f:
        content = f.read()
    print(content[:5000])
    if len(content) > 5000:
        print(f"\\n... (truncated, total {{len(content)}} chars)")
except FileNotFoundError:
    print(f"File not found: {{path}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
            result = forward(code=code, language="python", timeout=10)
            return result.get("stdout", "") or "File read failed."

        elif name == "task_complete":
            return f"TASK_COMPLETE: {arguments.get('summary', 'Done')}"

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {str(e)}"


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], operator.add]
    user_prompt: str
    dag: dict
    drs: dict
    current_task_idx: int
    ledger: dict
    difficulty_scores: dict


# ---------------------------------------------------------------------------
# Node 1: Decompose prompt into TaskDAG
# ---------------------------------------------------------------------------

def decompose_and_evaluate(state: AgentState):
    print("🤖 Decomposing request into Fillmore Frames...")
    dag = parse_prompt(state["user_prompt"])

    mem = TriGraphMemory.load()
    initial_scores = {}

    try:
        retriever = mem.as_vector_retriever(similarity_top_k=10)
        all_knowledge = retriever.retrieve(state["user_prompt"])
        knowledge_text = "\n".join([n.text for n in all_knowledge[:5]]) if all_knowledge else ""
    except Exception:
        knowledge_text = ""
        all_knowledge = []

    for task in dag.tasks:
        score = compute_base_difficulty(
            task_id=task.id,
            task_description=task.description,
            reranker_edges=all_knowledge,
            composition_graph=nx.DiGraph(),
            knowledge_context=knowledge_text,
        )
        tier = difficulty_to_tier(score)
        initial_scores[task.id] = {"score": score, "tier": tier}
        print(f"  📊 Task '{task.description[:40]}...' → {tier} (K={score:.2f})")

    return {
        "dag": dag.model_dump(),
        "drs": DRS(label="main").to_dict(),
        "current_task_idx": 0,
        "ledger": CompositionLedger().to_dict(),
        "difficulty_scores": initial_scores,
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} frames. Initial difficulty assessed."}]
    }


# ---------------------------------------------------------------------------
# Node 2: Execute single task with ITERATIVE TOOL-USE LOOP
# ---------------------------------------------------------------------------

MAX_TOOL_ITERATIONS = 15

def execute_single_task(state: AgentState):
    import orchestrator.tool_sandbox as tool_sandbox

    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task = tasks[idx]
    task_id = task.get("id", "")
    task_desc = task.get("description", "")

    # --- Deserialize state ---
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger_data = state.get("ledger", {})
    ledger = CompositionLedger.from_dict(ledger_data) if ledger_data else CompositionLedger()

    # --- Get difficulty tier ---
    scores = state.get("difficulty_scores", {})
    task_scores = scores.get(task_id, {})
    tier = task_scores.get("tier", "moderate")
    score = task_scores.get("score", 0.5)

    print(f"\n🧭 Executing Task {idx + 1}/{len(tasks)}: [{tier.upper()}] (K={score:.2f})")
    print(f"   '{task_desc[:60]}'")

    with ModalKeepAlive(tier=tier):
        mem = TriGraphMemory.load()

        # 1. EPISTEMIC RETRIEVAL
        try:
            retriever = mem.as_vector_retriever(similarity_top_k=5)
            knowledge_nodes = retriever.retrieve(task_desc)
        except Exception as e:
            print(f"  ⚠️ Retrieval failed: {e}")
            knowledge_nodes = []

        # 2. DRT: Augment with discourse referents
        desc = drs.augment_description(task_desc)
        knowledge_str = "\n".join([n.text for n in knowledge_nodes[:3]]) if knowledge_nodes else "None"

        # 3. FREGE: Capture pre-execution state
        try:
            before_state = WorkspaceState.capture(tool_sandbox)
        except Exception:
            before_state = WorkspaceState.empty()

        # ==============================================================
        # 4. ITERATIVE TOOL-USE LOOP
        #
        # The model gets tool definitions and calls them iteratively
        # until it signals task_complete or hits the iteration cap.
        #
        # llama-server's OpenAI-compatible API handles tool_calls
        # natively. We just need to:
        #   a) Send tools in the request (via llm_client → Modal → llama-server)
        #   b) Parse tool_calls from the response
        #   c) Execute them via tool_sandbox.forward()
        #   d) Feed results back as tool messages
        #   e) Loop
        # ==============================================================

        prior_results = []
        for msg in state.get("messages", []):
            if msg.get("role") == "assistant" and "Result" in msg.get("content", ""):
                prior_results.append(msg["content"][:200])

        system_prompt = f"""You are an autonomous coding agent. You MUST use the provided tools to complete the task.
Do NOT just describe what to do — actually DO it by calling tools.

Your workspace is at /workspace. Available tools:
- web_search: Find information online
- run_code: Execute Python or bash code
- write_file: Create/modify files
- read_file: Read existing files
- task_complete: Signal when done (REQUIRED)

You MUST call task_complete when finished.

Prior task results:
{chr(10).join(prior_results[-3:]) if prior_results else 'This is the first task.'}
"""

        task_prompt = f"""Task: {desc}

Retrieved Knowledge:
{knowledge_str}

Begin working. Use tools to accomplish this task."""

        loop_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        all_tool_outputs = []
        task_completed = False
        completion_summary = ""

        for iteration in range(MAX_TOOL_ITERATIONS):
            print(f"  🔄 Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS}")

            # Call model WITH tools via the fixed llm_client
            response = chat_completion(
                loop_messages,
                tier=tier,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
            )

            # Parse response — llama-server returns OpenAI ChatCompletion object
            response_message = _parse_response(response)
            loop_messages.append(response_message)

            # If no tool calls, handle text-only response
            if not response_message.get("tool_calls"):
                text_content = response_message.get("content", "")
                if text_content:
                    print(f"  💬 Model: {text_content[:100]}...")
                    all_tool_outputs.append(f"[Model]: {text_content[:500]}")

                if iteration == 0:
                    # Nudge on first iteration
                    loop_messages.append({
                        "role": "user",
                        "content": "Use the provided tools. Call web_search, run_code, write_file, or read_file. When done, call task_complete.",
                    })
                    continue
                else:
                    completion_summary = text_content[:500] if text_content else "Completed (no explicit summary)."
                    task_completed = True
                    break

            # Execute each tool call
            for tool_call in response_message.get("tool_calls", []):
                fn_name = tool_call["function"]["name"]
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                print(f"  🔧 Tool: {fn_name}({json.dumps(fn_args)[:80]}...)")

                result = _dispatch_tool(fn_name, fn_args)
                print(f"  📤 Result: {result[:120]}...")

                all_tool_outputs.append(f"[{fn_name}]: {result[:1000]}")

                if fn_name == "task_complete":
                    completion_summary = fn_args.get("summary", result)
                    task_completed = True
                    break

                # Feed result back to model as tool response
                loop_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                    "content": result[:2000],
                })

            if task_completed:
                break

        if not task_completed:
            completion_summary = f"Reached iteration limit ({MAX_TOOL_ITERATIONS}). Partial results."
            print(f"  ⚠️ {completion_summary}")

        exec_out = "\n".join(all_tool_outputs) if all_tool_outputs else "No outputs."

        # 5. FREGE: Post-execution state diff
        try:
            after_state = WorkspaceState.capture(tool_sandbox)
            delta = after_state.diff(before_state)
        except Exception:
            from orchestrator.composition import StateDelta
            delta = StateDelta(created=frozenset(), deleted=frozenset(), modified=frozenset())
            after_state = before_state

        ledger.record(task_id, before_state, after_state)

        # 6. DRT: Introduce new referents
        drs.introduce_from_delta(task_id, delta)

        # 7. EPISTEMIC UPDATE
        new_knowledge_text = ""
        if delta.created or delta.modified:
            print("  🧠 [Epistemic Update] Ingesting new artifacts...")
            build_knowledge_graphs("/workspace")
            try:
                mem = TriGraphMemory.load()
                retriever = mem.as_vector_retriever(similarity_top_k=10)
                new_knowledge = retriever.retrieve(state["user_prompt"])
                new_knowledge_text = "\n".join([n.text for n in new_knowledge[:5]])
            except Exception:
                pass

        # 8. REASSESS remaining tasks
        remaining_tasks = tasks[idx + 1:]
        updated_scores = dict(scores)

        if remaining_tasks:
            print("  📊 [Reassessment] Re-scoring remaining tasks...")
            try:
                comp_graph = ledger.derive_dependency_graph()
            except (ValueError, Exception):
                comp_graph = nx.DiGraph()

            reranker_edges = knowledge_nodes
            reassessed = reassess_remaining_tasks(
                remaining_tasks=remaining_tasks,
                knowledge_context=new_knowledge_text or "",
                reranker_edges=reranker_edges,
                composition_graph=comp_graph,
            )
            for tid, new_score, new_tier in reassessed:
                old = updated_scores.get(tid, {})
                old_tier = old.get("tier", "unknown")
                if new_tier != old_tier:
                    print(f"    ↕ Task {tid}: {old_tier} → {new_tier} (K={new_score:.2f})")
                updated_scores[tid] = {"score": new_score, "tier": new_tier}

        # 9. SUB-TASK SPAWNING
        spawned_tasks = _check_for_subtasks(exec_out, task, drs)

        dag_update = state.get("dag", {})
        if spawned_tasks:
            print(f"  🔀 Spawning {len(spawned_tasks)} sub-tasks...")
            current_tasks = dag_update.get("tasks", [])
            for i, st in enumerate(spawned_tasks):
                st_dict = st.model_dump()
                current_tasks.insert(idx + 1 + i, st_dict)
                sub_score = compute_base_difficulty(
                    task_id=st.id,
                    task_description=st.description,
                    reranker_edges=reranker_edges if remaining_tasks else [],
                    composition_graph=comp_graph if remaining_tasks else nx.DiGraph(),
                    knowledge_context=new_knowledge_text,
                )
                sub_tier = difficulty_to_tier(sub_score)
                updated_scores[st.id] = {"score": sub_score, "tier": sub_tier}
                print(f"    + '{st.description[:40]}...' → {sub_tier} (K={sub_score:.2f})")
            dag_update["tasks"] = current_tasks

    return {
        "dag": dag_update,
        "drs": drs.to_dict(),
        "current_task_idx": idx + 1,
        "ledger": ledger.to_dict(),
        "difficulty_scores": updated_scores,
        "messages": [{
            "role": "assistant",
            "content": (
                f"#### Task {idx + 1} ({tier}, K={score:.2f})\n"
                f"**Summary:** {completion_summary}\n\n"
                f"**Tool calls:** {len(all_tool_outputs)}\n"
                f"**Files changed:** {len(delta.created | delta.modified) if hasattr(delta, 'created') else 0}\n\n"
                f"**Output excerpt:**\n{exec_out[:500]}\n"
            )
        }]
    }


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def _parse_response(response) -> dict:
    """Normalize chat_completion response into a message dict.

    Handles the OpenAI ChatCompletion object returned by llama-server
    (via BaseInference.generate_logic when tools are present).
    """
    # OpenAI ChatCompletion object (from llama-server)
    if hasattr(response, "choices"):
        choice = response.choices[0]
        msg = {
            "role": "assistant",
            "content": getattr(choice.message, "content", None),
        }
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        return msg

    # Dict response (fallback)
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {
            "role": "assistant", "content": str(response)
        })

    # Raw string (old behavior fallback)
    return {"role": "assistant", "content": str(response)}


# ---------------------------------------------------------------------------
# Sub-task Spawning Heuristics
# ---------------------------------------------------------------------------

def _check_for_subtasks(exec_output: str, task: dict, drs: DRS) -> list:
    spawned = []
    output_lower = exec_output.lower() if exec_output else ""

    if any(kw in output_lower for kw in ["traceback", "error", "failed", "assertion"]):
        if task.get("task_type") != "verify":
            spawned.append(VerifyTask(
                description=f"Debug and fix error from: {task.get('description', '')[:80]}",
                tool_hint="python",
                preconditions=[task.get("id", "")],
            ))

    if "modulenotfounderror" in output_lower or "no module named" in output_lower:
        match = re.search(r"no module named ['\"]?(\w+)", output_lower)
        module_name = match.group(1) if match else "unknown"
        spawned.append(TransformTask(
            description=f"Install missing module '{module_name}' and retry",
            tool_hint="bash",
            preconditions=[task.get("id", "")],
        ))

    return spawned


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_next_step(state: AgentState) -> str:
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    if idx < len(tasks):
        return "execute_single_task"
    return "end"


# ---------------------------------------------------------------------------
# Graph Wiring
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_single_task", execute_single_task)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_single_task")
workflow.add_conditional_edges(
    "execute_single_task",
    route_next_step,
    {"execute_single_task": "execute_single_task", "end": END}
)
agent_executor = workflow.compile()


async def agent_loop_stream(prompt: str):
    initial_state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_prompt": prompt,
        "current_task_idx": 0,
        "difficulty_scores": {},
    }
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {
                "step": node,
                "state_snapshot": values,
                "node_update": values,
                "graphs": mem.to_dict(),
                "difficulty_scores": values.get("difficulty_scores", {}),
            }
    drs: dict
    current_task_idx: int
    ledger: dict
    # NEW: track reassessed difficulty scores for remaining tasks
    difficulty_scores: dict  # {task_id: (score, tier)}
# ---------------------------------------------------------------------------
# Node 1: Decompose prompt into TaskDAG
# ---------------------------------------------------------------------------
def decompose_and_evaluate(state: AgentState):
    """
    FILLMORE FRAME EXTRACTION:
    Translates natural language into strict, typed operational frames.
    Outputs a DAG (not linear chain) with dependency edges.
    """
    print("🤖 Decomposing request into Fillmore Frames...")
    dag = parse_prompt(state["user_prompt"])
    # Pre-compute initial difficulty for all tasks
    mem = TriGraphMemory.load()
    initial_scores = {}
    # Build knowledge context from current graph
    try:
        retriever = mem.as_vector_retriever(similarity_top_k=10)
        all_knowledge = retriever.retrieve(state["user_prompt"])
        knowledge_text = "\n".join([n.text for n in all_knowledge[:5]]) if all_knowledge else ""
    except Exception:
        knowledge_text = ""
        all_knowledge = []
    for task in dag.tasks:
        score = compute_base_difficulty(
            task_id=task.id,
            task_description=task.description,
            reranker_edges=all_knowledge,
            composition_graph=nx.DiGraph(),
            knowledge_context=knowledge_text,
        )
        tier = difficulty_to_tier(score)
        initial_scores[task.id] = {"score": score, "tier": tier}
        print(f"  📊 Task '{task.description[:40]}...' → {tier} (K={score:.2f})")
    return {
        "dag": dag.model_dump(),
        "drs": DRS(label="main").to_dict(),
        "current_task_idx": 0,
        "ledger": CompositionLedger().to_dict(),
        "difficulty_scores": initial_scores,
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} frames. Initial difficulty assessed."}]
    }
# ---------------------------------------------------------------------------
# Node 2: Execute single task with full feedback loop
# ---------------------------------------------------------------------------
def execute_single_task(state: AgentState):
    from orchestrator import tool_sandbox
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task = tasks[idx]
    task_id = task.get("id", "")
    task_desc = task.get("description", "")
    # --- Deserialize state (FIX: was using non-existent .events attribute) ---
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger_data = state.get("ledger", {})
    ledger = CompositionLedger.from_dict(ledger_data) if ledger_data else CompositionLedger()
    # --- Get difficulty tier (may have been reassessed by previous task) ---
    scores = state.get("difficulty_scores", {})
    task_scores = scores.get(task_id, {})
    tier = task_scores.get("tier", "moderate")
    score = task_scores.get("score", 0.5)
    print(f"\n🧭 Executing Task {idx + 1}/{len(tasks)}: [{tier.upper()}] (K={score:.2f})")
    print(f"   '{task_desc[:60]}'")
    with ModalKeepAlive(tier=tier):
        mem = TriGraphMemory.load()
        # 1. EPISTEMIC RETRIEVAL (RAG)
        try:
            retriever = mem.as_vector_retriever(similarity_top_k=5)
            knowledge_nodes = retriever.retrieve(task_desc)
        except Exception as e:
            print(f"  ⚠️ Retrieval failed (graph may be empty): {e}")
            knowledge_nodes = []
        # 2. DRT: Augment description with resolved discourse referents
        desc = drs.augment_description(task_desc)
        prompt = f"Implement task. Context:\n{desc}\n\nRetrieved Knowledge:\n"
        prompt += "\n".join([n.text for n in knowledge_nodes[:3]]) if knowledge_nodes else "None"
        # 3. FREGE: Capture pre-execution state
        try:
            before_state = WorkspaceState.capture(tool_sandbox)
        except Exception:
            before_state = WorkspaceState.empty()
        # 4. GENERATION: Route to appropriate model tier
        res = chat_completion(
            [{"role": "user", "content": prompt}],
            tier=tier,  # This should select the model based on difficulty
        )
        # 5. EXECUTE: Run generated code in sandbox
        exec_out = execute_code(str(res))
        # 6. FREGE: Capture post-execution state and diff
        try:
            after_state = WorkspaceState.capture(tool_sandbox)
            delta = after_state.diff(before_state)
        except Exception:
            from orchestrator.composition import StateDelta
            delta = StateDelta(created=frozenset(), deleted=frozenset(), modified=frozenset())
            after_state = before_state
        # Record in composition ledger
        ledger.record(task_id, before_state, after_state)
        # 7. DRT: Introduce new referents from filesystem changes
        drs.introduce_from_delta(task_id, delta)
        # 8. EPISTEMIC UPDATE: Re-index if files changed
        new_knowledge_text = ""
        if delta.created or delta.modified:
            print("  🧠 [Epistemic Update] Ingesting new artifacts into Tri-Graph...")
            build_knowledge_graphs("/workspace")
            # Refresh knowledge context for reassessment
            try:
                mem = TriGraphMemory.load()
                retriever = mem.as_vector_retriever(similarity_top_k=10)
                new_knowledge = retriever.retrieve(state["user_prompt"])
                new_knowledge_text = "\n".join([n.text for n in new_knowledge[:5]])
            except Exception:
                pass
        # 9. REASSESS remaining tasks' difficulty
        remaining_tasks = tasks[idx + 1:]
        updated_scores = dict(scores)  # Copy existing scores
        if remaining_tasks:
            print("  📊 [Reassessment] Re-scoring remaining tasks against updated graph...")
            try:
                comp_graph = ledger.derive_dependency_graph()
            except (ValueError, Exception):
                comp_graph = nx.DiGraph()
            # Get fresh reranker edges
            reranker_edges = knowledge_nodes  # Use latest retrieval
            reassessed = reassess_remaining_tasks(
                remaining_tasks=remaining_tasks,
                knowledge_context=new_knowledge_text or "",
                reranker_edges=reranker_edges,
                composition_graph=comp_graph,
            )
            for tid, new_score, new_tier in reassessed:
                old = updated_scores.get(tid, {})
                old_tier = old.get("tier", "unknown")
                if new_tier != old_tier:
                    print(f"    ↕ Task {tid}: {old_tier} → {new_tier} (K={new_score:.2f})")
                updated_scores[tid] = {"score": new_score, "tier": new_tier}
        # 10. SUB-TASK SPAWNING: Check if execution revealed new complexity
        spawned_tasks = _check_for_subtasks(exec_out, task, drs)
        dag_update = state.get("dag", {})
        if spawned_tasks:
            print(f"  🔀 Spawning {len(spawned_tasks)} sub-tasks from execution results...")
            # Insert sub-tasks right after current task
            current_tasks = dag_update.get("tasks", [])
            for i, st in enumerate(spawned_tasks):
                st_dict = st.model_dump()
                current_tasks.insert(idx + 1 + i, st_dict)
                # Score the new sub-task
                sub_score = compute_base_difficulty(
                    task_id=st.id,
                    task_description=st.description,
                    reranker_edges=reranker_edges if remaining_tasks else [],
                    composition_graph=comp_graph if remaining_tasks else nx.DiGraph(),
                    knowledge_context=new_knowledge_text,
                )
                sub_tier = difficulty_to_tier(sub_score)
                updated_scores[st.id] = {"score": sub_score, "tier": sub_tier}
                print(f"    + Sub-task '{st.description[:40]}...' → {sub_tier} (K={sub_score:.2f})")
            dag_update["tasks"] = current_tasks
    return {
        "dag": dag_update,
        "drs": drs.to_dict(),
        "current_task_idx": idx + 1,
        "ledger": ledger.to_dict(),
        "difficulty_scores": updated_scores,
        "messages": [{
            "role": "assistant",
            "content": f"#### Task {idx + 1} ({tier}, K={score:.2f})\n**Result:**\n{exec_out}\n\n"
        }]
    }
# ---------------------------------------------------------------------------
# Sub-task Spawning Heuristics
# ---------------------------------------------------------------------------
def _check_for_subtasks(exec_output: str, task: dict, drs: DRS) -> list:
    """Detect if execution output indicates need for sub-tasks.
    Heuristics (deterministic, no LLM needed):
    1. Test failures → spawn a debug + re-test task
    2. Import errors → spawn a research + install task
    3. Multiple files created → spawn verify tasks for each
    This is where the system self-corrects without human intervention.
    The heuristics are intentionally conservative — only spawn when
    there's strong signal.
    """
    spawned = []
    output_lower = exec_output.lower() if exec_output else ""
    # Heuristic 1: Test failure → debug + re-test
    if any(kw in output_lower for kw in ["traceback", "error", "failed", "assertion"]):
        if task.get("task_type") != "verify":
            # Don't spawn infinite verify loops
            spawned.append(VerifyTask(
                description=f"Debug and fix error from: {task.get('description', '')[:80]}",
                tool_hint="python",
                preconditions=[task.get("id", "")],
            ))
    # Heuristic 2: Import error → research + install
    if "modulenotfounderror" in output_lower or "no module named" in output_lower:
        # Extract module name (best effort)
        import re
        match = re.search(r"no module named ['\"]?(\w+)", output_lower)
        module_name = match.group(1) if match else "unknown"
        from orchestrator.task_parser import TransformTask as TT
        spawned.append(TT(
            description=f"Install missing module '{module_name}' and retry",
            tool_hint="bash",
            preconditions=[task.get("id", "")],
        ))
    return spawned
# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_next_step(state: AgentState) -> str:
    idx = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    if idx < len(tasks):
        return "execute_single_task"
    return "end"
# ---------------------------------------------------------------------------
# Graph Wiring
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_and_evaluate)
workflow.add_node("execute_single_task", execute_single_task)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute_single_task")
workflow.add_conditional_edges(
    "execute_single_task",
    route_next_step,
    {"execute_single_task": "execute_single_task", "end": END}
)
agent_executor = workflow.compile()
async def agent_loop_stream(prompt: str):
    initial_state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_prompt": prompt,
        "current_task_idx": 0,
        "difficulty_scores": {},
    }
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {
                "step": node,
                "state_snapshot": values,
                "node_update": values,
                "graphs": mem.to_dict(),
                "difficulty_scores": values.get("difficulty_scores", {}),
            }
