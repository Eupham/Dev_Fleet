"""LangGraph agent loop — iterative tool-use execution, difficulty reassessment,
and sub-task spawning.

Node names match what ui/web.py renders:
  "Decompose"           -> decompose_and_evaluate()
  "execute_single_task" -> execute_single_task()

Each yielded update dict contains:
  step              : node name
  state_snapshot    : full LangGraph state after node
  node_update       : same dict (partial update returned by node)
  graphs            : serialised TriGraphMemory
  model_info        : {tier, model, gpu, class_name} for the current task
  sandbox_results   : list of {tool, args, output} records for every tool call
"""

import operator
import json
import re
from typing import Annotated, List, TypedDict

from langgraph.graph import StateGraph, END
import networkx as nx

# All tool primitives, schema and dispatch live in tool_sandbox (SoC).
from orchestrator.tool_sandbox import (
    AVAILABLE_TOOLS,
    dispatch_tool,
)
from orchestrator.task_parser import parse_prompt, TransformTask, VerifyTask
from orchestrator.composition import WorkspaceState, CompositionLedger
from orchestrator.discourse import DRS, ReferentType  # noqa: F401
from orchestrator.llm_client import chat_completion, ModalKeepAlive
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.difficulty import (
    compute_base_difficulty,
    difficulty_to_tier,
    reassess_remaining_tasks,
)
from orchestrator.indexer import build_knowledge_graphs


# ---------------------------------------------------------------------------
# Model / GPU metadata table (mirrors inference/config.toml + model_pool.py)
# ---------------------------------------------------------------------------

_TIER_META = {
    "trivial":  {"class_name": "InferenceSmall",  "model": "Qwen3.5-4B-Q4_K_M",  "gpu": "T4"},
    "simple":   {"class_name": "InferenceMedium", "model": "Qwen3.5-9B-Q4_K_M",  "gpu": "L4"},
    "moderate": {"class_name": "Inference",        "model": "Qwen3.5-35B-A3B-Q4_K_M", "gpu": "L40S"},
    "complex":  {"class_name": "Inference",        "model": "Qwen3.5-35B-A3B-Q4_K_M", "gpu": "L40S"},
    "expert":   {"class_name": "InferenceLarge",   "model": "Qwen3.5-35B-A3B-Q4_K_M", "gpu": "L40S"},
}


def _model_info(tier: str) -> dict:
    meta = _TIER_META.get(tier, _TIER_META["moderate"])
    return {"tier": tier, **meta}


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    messages:          Annotated[List[dict], operator.add]
    user_prompt:       str
    dag:               dict
    drs:               dict
    current_task_idx:  int
    ledger:            dict
    difficulty_scores: dict
    # Surfaced to the UI per-step
    model_info:        dict   # {tier, model, gpu, class_name}
    sandbox_results:   list   # [{tool, args, output}, ...]


# ---------------------------------------------------------------------------
# Node 1: Decompose
# ---------------------------------------------------------------------------

def decompose_and_evaluate(state: AgentState) -> dict:
    print("Decomposing request into Fillmore Frames...")
    dag = parse_prompt(state["user_prompt"])

    mem = TriGraphMemory.load()
    initial_scores: dict = {}

    try:
        retriever = mem.as_vector_retriever(similarity_top_k=10)
        all_knowledge = retriever.retrieve(state["user_prompt"])
        knowledge_text = (
            "\n".join([n.text for n in all_knowledge[:5]])
            if all_knowledge else ""
        )
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
        print(f"  Task '{task.description[:40]}...' -> {tier} (K={score:.2f})")

    return {
        "dag":               dag.model_dump(),
        "drs":               DRS(label="main").to_dict(),
        "current_task_idx":  0,
        "ledger":            CompositionLedger().to_dict(),
        "difficulty_scores": initial_scores,
        "sandbox_results":   [],
        "model_info":        {},
        "messages": [{
            "role": "assistant",
            "content": f"Planned {len(dag.tasks)} frames. Initial difficulty assessed.",
        }],
    }


# ---------------------------------------------------------------------------
# Node 2: Execute single task — iterative tool-use loop
# ---------------------------------------------------------------------------

MAX_TOOL_ITERATIONS = 15


def execute_single_task(state: AgentState) -> dict:
    import orchestrator.tool_sandbox as tool_sandbox_module

    idx   = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    task  = tasks[idx]

    task_id   = task.get("id", "")
    task_desc = task.get("description", "")

    # --- Deserialize state ---
    drs = DRS.from_dict(state.get("drs", {"label": "main"}))
    ledger_data = state.get("ledger", {})
    ledger = CompositionLedger.from_dict(ledger_data) if ledger_data else CompositionLedger()

    # --- Difficulty tier ---
    scores     = state.get("difficulty_scores", {})
    task_meta  = scores.get(task_id, {})
    tier       = task_meta.get("tier", "moderate")
    score      = task_meta.get("score", 0.5)
    info       = _model_info(tier)

    print(f"\nExecuting Task {idx + 1}/{len(tasks)}: [{tier.upper()}] "
          f"model={info['model']} gpu={info['gpu']} K={score:.2f}")
    print(f"  '{task_desc[:60]}'")

    with ModalKeepAlive(tier=tier):
        mem = TriGraphMemory.load()

        # 1. Epistemic retrieval
        try:
            retriever = mem.as_vector_retriever(similarity_top_k=5)
            knowledge_nodes = retriever.retrieve(task_desc)
        except Exception as e:
            print(f"  Retrieval failed: {e}")
            knowledge_nodes = []

        # 2. DRT augmentation
        desc          = drs.augment_description(task_desc)
        knowledge_str = (
            "\n".join([n.text for n in knowledge_nodes[:3]])
            if knowledge_nodes else "None"
        )

        # 3. Pre-execution workspace snapshot
        try:
            before_state = WorkspaceState.capture(tool_sandbox_module)
        except Exception:
            before_state = WorkspaceState.empty()

        # 4. Build prior-results context for system prompt
        prior_results = []
        for msg in state.get("messages", []):
            if msg.get("role") == "assistant" and "Result" in msg.get("content", ""):
                prior_results.append(msg["content"][:200])

        system_prompt = (
            "You are an autonomous coding agent. You MUST use the provided tools to complete the task.\n"
            "Do NOT describe what to do — actually DO it by calling tools.\n\n"
            "Your workspace is at /workspace. Available tools:\n"
            "- web_search: Find information online\n"
            "- run_code: Execute Python or bash code\n"
            "- write_file: Create/modify files\n"
            "- read_file: Read existing files\n"
            "- task_complete: Signal when done (REQUIRED)\n\n"
            "You MUST call task_complete when finished.\n\n"
            f"Prior task results:\n"
            + ("\n".join(prior_results[-3:]) if prior_results else "This is the first task.")
        )

        task_prompt = (
            f"Task: {desc}\n\n"
            f"Retrieved Knowledge:\n{knowledge_str}\n\n"
            "Begin working. Use tools to accomplish this task."
        )

        loop_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": task_prompt},
        ]

        # sandbox_results accumulates every tool call for the UI
        sandbox_results: list = []
        task_completed      = False
        completion_summary  = ""

        # 5. Iterative tool-use loop
        for iteration in range(MAX_TOOL_ITERATIONS):
            print(f"  Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS}")

            response = chat_completion(
                loop_messages,
                tier=tier,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
            )

            response_message = _parse_response(response)
            loop_messages.append(response_message)

            tool_calls = response_message.get("tool_calls") or []

            # No tool calls — model returned plain text
            if not tool_calls:
                text_content = response_message.get("content", "")
                if text_content:
                    print(f"  Model: {text_content[:100]}...")
                    sandbox_results.append({
                        "tool":   "[model_text]",
                        "args":   {},
                        "output": text_content[:1000],
                    })

                if iteration == 0:
                    loop_messages.append({
                        "role":    "user",
                        "content": (
                            "Use the provided tools. Call web_search, run_code, "
                            "write_file, or read_file. When done, call task_complete."
                        ),
                    })
                    continue
                else:
                    completion_summary = text_content[:500] if text_content else "Completed."
                    task_completed = True
                    break

            # Execute every tool call in this turn
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                print(f"  Tool: {fn_name}({json.dumps(fn_args)[:80]}...)")

                # Dispatch via tool_sandbox (single authoritative router)
                result = dispatch_tool(fn_name, fn_args)
                print(f"  Result: {result[:120]}...")

                sandbox_results.append({
                    "tool":   fn_name,
                    "args":   fn_args,
                    "output": result[:2000],
                })

                if fn_name == "task_complete":
                    completion_summary = fn_args.get("summary", result)
                    task_completed = True
                    break

                loop_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", f"call_{iteration}"),
                    "content":      result[:2000],
                })

            if task_completed:
                break

        if not task_completed:
            completion_summary = f"Reached iteration limit ({MAX_TOOL_ITERATIONS}). Partial results."
            print(f"  {completion_summary}")

        exec_out = "\n".join(
            f"[{r['tool']}]: {r['output']}" for r in sandbox_results
        ) if sandbox_results else "No outputs."

        # 6. Post-execution workspace diff
        try:
            after_state = WorkspaceState.capture(tool_sandbox_module)
            delta       = after_state.diff(before_state)
        except Exception:
            from orchestrator.composition import StateDelta
            delta       = StateDelta(created=frozenset(), deleted=frozenset(), modified=frozenset())
            after_state = before_state

        ledger.record(task_id, before_state, after_state)
        drs.introduce_from_delta(task_id, delta)

        # 7. Epistemic update — ingest new files into knowledge graph
        new_knowledge_text = ""
        if delta.created or delta.modified:
            print("  [Epistemic Update] Ingesting new artifacts...")
            build_knowledge_graphs("/workspace")
            try:
                mem       = TriGraphMemory.load()
                retriever = mem.as_vector_retriever(similarity_top_k=10)
                new_knowledge = retriever.retrieve(state["user_prompt"])
                new_knowledge_text = "\n".join([n.text for n in new_knowledge[:5]])
            except Exception:
                pass

        # 8. Reassess remaining tasks
        remaining_tasks = tasks[idx + 1:]
        updated_scores  = dict(scores)

        if remaining_tasks:
            print("  [Reassessment] Re-scoring remaining tasks...")
            try:
                comp_graph = ledger.derive_dependency_graph()
            except Exception:
                comp_graph = nx.DiGraph()

            reassessed = reassess_remaining_tasks(
                remaining_tasks=remaining_tasks,
                knowledge_context=new_knowledge_text or "",
                reranker_edges=knowledge_nodes,
                composition_graph=comp_graph,
            )
            for tid, new_score, new_tier in reassessed:
                old_tier = updated_scores.get(tid, {}).get("tier", "unknown")
                if new_tier != old_tier:
                    print(f"    Task {tid}: {old_tier} -> {new_tier} (K={new_score:.2f})")
                updated_scores[tid] = {"score": new_score, "tier": new_tier}

        # 9. Sub-task spawning
        spawned_tasks = _check_for_subtasks(exec_out, task, drs)
        dag_update    = state.get("dag", {})

        if spawned_tasks:
            print(f"  Spawning {len(spawned_tasks)} sub-tasks...")
            current_tasks = dag_update.get("tasks", [])
            try:
                comp_graph = ledger.derive_dependency_graph()
            except Exception:
                comp_graph = nx.DiGraph()

            for i, st in enumerate(spawned_tasks):
                st_dict = st.model_dump()
                current_tasks.insert(idx + 1 + i, st_dict)
                sub_score = compute_base_difficulty(
                    task_id=st.id,
                    task_description=st.description,
                    reranker_edges=knowledge_nodes,
                    composition_graph=comp_graph,
                    knowledge_context=new_knowledge_text,
                )
                sub_tier = difficulty_to_tier(sub_score)
                updated_scores[st.id] = {"score": sub_score, "tier": sub_tier}
                print(f"    + '{st.description[:40]}...' -> {sub_tier} (K={sub_score:.2f})")
            dag_update["tasks"] = current_tasks

    return {
        "dag":               dag_update,
        "drs":               drs.to_dict(),
        "current_task_idx":  idx + 1,
        "ledger":            ledger.to_dict(),
        "difficulty_scores": updated_scores,
        "model_info":        info,
        "sandbox_results":   sandbox_results,
        "messages": [{
            "role": "assistant",
            "content": (
                f"#### Task {idx + 1} ({tier}, K={score:.2f})\n"
                f"**Model:** {info['model']} on {info['gpu']}\n"
                f"**Summary:** {completion_summary}\n\n"
                f"**Tool calls:** {len(sandbox_results)}\n\n"
                f"**Output excerpt:**\n{exec_out[:500]}\n"
            ),
        }],
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(response) -> dict:
    """Normalize a chat_completion response to a plain message dict."""
    if hasattr(response, "choices"):
        choice = response.choices[0]
        msg = {
            "role":    "assistant",
            "content": getattr(choice.message, "content", None),
        }
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        return msg

    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {
            "role": "assistant", "content": str(response)
        })

    return {"role": "assistant", "content": str(response)}


# ---------------------------------------------------------------------------
# Sub-task spawning heuristics
# ---------------------------------------------------------------------------

def _check_for_subtasks(exec_output: str, task: dict, drs: DRS) -> list:
    spawned     = []
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
    idx   = state.get("current_task_idx", 0)
    tasks = state.get("dag", {}).get("tasks", [])
    return "execute_single_task" if idx < len(tasks) else "end"


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

# Node names are the step names ui/web.py checks in its elif ladder.
workflow = StateGraph(AgentState)
workflow.add_node("Decompose",          decompose_and_evaluate)
workflow.add_node("execute_single_task", execute_single_task)
workflow.set_entry_point("Decompose")
workflow.add_edge("Decompose", "execute_single_task")
workflow.add_conditional_edges(
    "execute_single_task",
    route_next_step,
    {"execute_single_task": "execute_single_task", "end": END},
)
agent_executor = workflow.compile()


# ---------------------------------------------------------------------------
# Public streaming entrypoint
# ---------------------------------------------------------------------------

async def agent_loop_stream(prompt: str):
    initial_state = {
        "messages":          [{"role": "user", "content": prompt}],
        "user_prompt":       prompt,
        "current_task_idx":  0,
        "difficulty_scores": {},
        "sandbox_results":   [],
        "model_info":        {},
    }
    async for event in agent_executor.astream(initial_state):
        for node, values in event.items():
            mem = TriGraphMemory.load()
            yield {
                "step":             node,
                "state_snapshot":   values,
                "node_update":      values,
                "graphs":           mem.to_dict(),
                "model_info":       values.get("model_info", {}),
                "sandbox_results":  values.get("sandbox_results", []),
                "difficulty_scores": values.get("difficulty_scores", {}),
            }
