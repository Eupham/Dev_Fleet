"""LangGraph agent loop — iterative tool-use with fallback resilience.

FIXES APPLIED (v3 — against actual repo state):
1. EXECUTION WAS A NO-OP: Replaced single-shot chat_completion → execute_code(str(res))
   with iterative tool-use loop (up to 15 iterations).

2. ERROR RESILIENCE: The critical crash was:
   - llama-server returns 500 when model generates malformed tool-call JSON
     (e.g. a 13k-char Python script as a raw argument)
   - openai.APIStatusError can't be deserialized across Modal RPC boundary
     (missing 'response'/'body' kwargs)
   Fixed: try/except around chat_completion with retry + fallback to no-tools mode.
   Matching fix in utils.py wraps exceptions in RuntimeError for serialization.

3. RESPONSE PARSING: Handles BOTH:
   - dict responses (from utils.py v3 which serializes ChatCompletion to dict)
   - OpenAI ChatCompletion objects (legacy/if utils.py hasn't been updated)
   - plain strings (fallback)

4. LOG NOISE FILTER: Suppresses CUDA boilerplate, server pings, routine HTTP,
   HF auth warnings, and truncates massive payloads in error messages.

5. LEDGER BUG: Fixed CompositionLedger deserialization.
6. DIFFICULTY REASSESSMENT: Re-scores remaining tasks after each execution.
7. SUB-TASK SPAWNING: Deterministic heuristics for error recovery.
"""

import operator
import os
import json
import re
import logging
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

logger = logging.getLogger("dev_fleet.orchestrator")


# ---------------------------------------------------------------------------
# Log Noise Filter — silences CUDA boilerplate, server pings, etc.
# ---------------------------------------------------------------------------
class _NoiseFilter(logging.Filter):
    NOISE = [
        "CUDA Version", "Container image Copyright", "NVIDIA Deep Learning",
        "ggml_cuda_init", "Booting raw llama-server", "Server is online",
        "GET /user ->", "GET /project/translations", "GET /ws/socket.io",
        "POST /ws/socket.io", "CONNECT /ws/socket.io",
        "Received second interrupt", "unauthenticated requests to the HF Hub",
        "Loading weights:", "NGC-DL-CONTAINER-LICENSE",
    ]
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self.NOISE)

logging.getLogger().addFilter(_NoiseFilter())


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
    """Execute a tool call via tool_sandbox.forward()."""
    try:
        if name == "web_search":
            query = arguments.get("query", "")
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
            return (result.get("stdout", "") or "No search results.")[:2000]

        elif name == "run_code":
            language = arguments.get("language", "python")
            code = arguments.get("code", "")
            result = forward(code=code, language=language, timeout=30)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            output = stdout or "Code executed (no stdout)."
            if stderr:
                output += f"\nSTDERR: {stderr[:500]}"
            return output[:2000]

        elif name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
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
        return f"Tool error ({name}): {str(e)[:500]}"


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
        print(f"  📊 Task '{task.description[:50]}...' → {tier} (K={score:.2f})")

    return {
        "dag": dag.model_dump(),
        "drs": DRS(label="main").to_dict(),
        "current_task_idx": 0,
        "ledger": CompositionLedger().to_dict(),
        "difficulty_scores": initial_scores,
        "messages": [{"role": "assistant", "content": f"Planned {len(dag.tasks)} frames. Initial difficulty assessed."}]
    }


# ---------------------------------------------------------------------------
# Response Parsing — handles dict (v3), OpenAI objects, and strings
# ---------------------------------------------------------------------------

def _parse_response(response) -> dict:
    """Normalize chat_completion response into a message dict.

    Handles:
    - dict with choices[0].message (from utils.py v3 serialization)
    - OpenAI ChatCompletion object (legacy, if utils.py not updated)
    - plain string (old behavior)
    """
    # Dict response (from utils.py v3 _serialize_response / _fallback_tool_generation)
    if isinstance(response, dict):
        if "choices" in response:
            msg = response["choices"][0].get("message", {})
            result = {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content"),
            }
            if msg.get("tool_calls"):
                result["tool_calls"] = msg["tool_calls"]
            return result
        # Unknown dict format
        return {"role": "assistant", "content": str(response)}

    # OpenAI ChatCompletion object (legacy)
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

    # Raw string fallback
    return {"role": "assistant", "content": str(response)}


# ---------------------------------------------------------------------------
# Node 2: Execute single task with ITERATIVE TOOL-USE LOOP
# ---------------------------------------------------------------------------

MAX_TOOL_ITERATIONS = 15
MAX_CONSECUTIVE_ERRORS = 3

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
            logger.warning(f"Retrieval failed: {e}")
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

IMPORTANT: Keep tool arguments concise. Do NOT put entire programs in a single argument.
Break large code into multiple write_file + run_code calls.
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
        consecutive_errors = 0
        use_tools = True  # Can be disabled after repeated failures

        for iteration in range(MAX_TOOL_ITERATIONS):
            print(f"  🔄 Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS}")

            # --- Call model with error resilience ---
            try:
                response = chat_completion(
                    loop_messages,
                    tier=tier,
                    tools=AVAILABLE_TOOLS if use_tools else None,
                    tool_choice="auto" if use_tools else None,
                )
                consecutive_errors = 0
            except RuntimeError as e:
                consecutive_errors += 1
                error_msg = str(e)[:200]
                print(f"  ⚠️ Inference error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {error_msg}")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    if use_tools:
                        # FALLBACK: Disable native tools, rely on structured prompt
                        print("  🔄 Disabling native tools, switching to structured prompt fallback...")
                        use_tools = False
                        consecutive_errors = 0
                        loop_messages.append({
                            "role": "user",
                            "content": (
                                "The tool-calling system had errors. Instead, respond with a JSON block "
                                "like: ```json\n{\"tool\": \"run_code\", \"arguments\": {\"code\": \"print('hello')\", \"language\": \"python\"}}\n``` "
                                "to use tools. Keep code SHORT."
                            ),
                        })
                        continue
                    else:
                        # Total failure
                        completion_summary = f"FAILED after {MAX_CONSECUTIVE_ERRORS} consecutive errors: {error_msg}"
                        print(f"  ❌ Aborting task: {error_msg}")
                        break

                # Simple retry with hint
                loop_messages.append({
                    "role": "user",
                    "content": "Previous attempt failed. Try again with shorter, simpler tool arguments.",
                })
                continue

            # --- Parse response ---
            response_message = _parse_response(response)
            loop_messages.append(response_message)

            # If no tool calls, check for embedded JSON tool calls (fallback mode)
            if not response_message.get("tool_calls"):
                text_content = response_message.get("content", "") or ""

                # Try to extract embedded JSON tool call
                embedded = _extract_embedded_tool_call(text_content)
                if embedded:
                    response_message["tool_calls"] = [embedded]

            if not response_message.get("tool_calls"):
                text_content = response_message.get("content", "") or ""
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
                    # Model stopped using tools — check if it produced code
                    if "```" in text_content:
                        code_result = execute_code(text_content)
                        if code_result:
                            all_tool_outputs.append(f"[code_extract]: {code_result[:500]}")
                    completion_summary = text_content[:500] if text_content else "Completed (no explicit summary)."
                    task_completed = True
                    break

            # Execute each tool call
            for tool_call in response_message.get("tool_calls", []):
                fn_name = tool_call.get("function", {}).get("name", "unknown")
                raw_args = tool_call.get("function", {}).get("arguments", "{}")
                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    fn_args = {"raw": str(raw_args)[:500]}
                    print(f"  ⚠️ Malformed tool args for {fn_name}, using raw fallback")

                print(f"  🔧 Tool: {fn_name}({json.dumps(fn_args)[:100]}...)")

                result = _dispatch_tool(fn_name, fn_args)
                print(f"  📤 Result: {result[:120]}...")

                all_tool_outputs.append(f"[{fn_name}]: {result[:1000]}")

                if fn_name == "task_complete":
                    completion_summary = fn_args.get("summary", result)
                    task_completed = True
                    break

                # Feed result back to model
                loop_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                    "content": result[:2000],
                })

            if task_completed:
                break

        if not task_completed:
            completion_summary = completion_summary or f"Reached iteration limit ({MAX_TOOL_ITERATIONS}). Partial results."
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


def _extract_embedded_tool_call(content: str) -> dict | None:
    """Extract a JSON tool call from model text when native tool-calling wasn't used."""
    # Try ```json blocks first
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if "tool" in parsed:
                return {
                    "id": "embedded_0",
                    "type": "function",
                    "function": {
                        "name": parsed["tool"],
                        "arguments": json.dumps(parsed.get("arguments", {}))
                    }
                }
        except json.JSONDecodeError:
            pass
    return None


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