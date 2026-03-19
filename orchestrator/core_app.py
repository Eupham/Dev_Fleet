"""Orchestrator — Tri-Graph agent loop (CPU container).

Coordinates graph memory, Frege parsing, Qwen3-Reranker scoring,
sandbox execution, and the main agent loop.  All inference calls
use Modal-native RPC to the Inference and Reranker classes within
the same ``dev_fleet`` app — no HTTP overhead or idle timeouts.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Any

import modal

from fleet_app import app  # shared app defined in app.py

# ---------------------------------------------------------------------------
# Logging — all critical state goes to stdout for `modal app logs`
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("dev_fleet.orchestrator")

# ---------------------------------------------------------------------------
# Container image — lightweight CPU image for orchestration logic
# ---------------------------------------------------------------------------

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("fleet_app", copy=True)
    .add_local_python_source("inference", copy=True)
    .add_local_python_source("orchestrator", copy=True)
    .apt_install("libblas-dev", "liblapack-dev")
    .uv_pip_install(
        "scipy<1.16.0",
        "networkx>=3.2",
        "pydantic==2.11.10",
        "fastapi==0.115.14",
        "pathspec>=0.12.1",
        "orjson>=3.11.7",
        "llama-index-core>=0.10.0",
        "llama-index>=0.10.0",
        "llama-index-embeddings-huggingface>=0.1.0",
        "langgraph>=1.1.2",
        "mcp>=1.26.0",
        "fastmcp==2.10.6",
        "radon>=6.0",
        "tree-sitter>=0.25.2",
        "tree-sitter-javascript>=0.23",
        "markdown-it-py>=3.0",
        "beautifulsoup4>=4.12",
        "trafilatura>=2.0.0",
        "pymupdf>=1.27.2",
        "ddgs>=6.3.7",           # web_search tool — baked in so no runtime install
        "playwright>=1.49.1",
    )
    .run_commands("NODE_OPTIONS='--no-deprecation' playwright install --with-deps chromium")
)

# ---------------------------------------------------------------------------
# Volumes / state persistence
# ---------------------------------------------------------------------------

graph_state_vol = modal.Volume.from_name(
    "dev_fleet-graph-state", create_if_missing=True
)
workspace_vol = modal.Volume.from_name(
    "dev_fleet-workspace", create_if_missing=True
)


# ---------------------------------------------------------------------------
# Structured logging helper
# ---------------------------------------------------------------------------


def log_event(event_type: str, payload: dict | None = None) -> None:
    """Emit a structured JSON log line for ``modal app logs`` consumption."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **(payload or {}),
    }
    logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Orchestrator entrypoint — invoked via `modal run app.py`
# ---------------------------------------------------------------------------


@app.function(
    image=orchestrator_image,
    volumes={
        "/state": graph_state_vol,
        "/workspace": workspace_vol,
    },
    timeout=30 * 60,
    # Generators in Modal cannot have retries
)
async def run_agent_stream(user_prompt: str) -> AsyncIterator[Dict[str, Any]]:
    """Execute a single agent loop iteration for *user_prompt* as an async generator.

    Yields intermediate states of the LangGraph execution to allow
    real-time rendering in the Chainlit UI.

    Parameters
    ----------
    user_prompt:
        The natural-language task from the user.

    Yields
    -------
    dict — intermediate states mapping step -> output + graph_state
    """
    from orchestrator.agent_loop import agent_loop_stream  # local import inside container

    log_event("agent_start", {"prompt": user_prompt[:200]})

    try:
        async for update in agent_loop_stream(user_prompt):
            yield update
        log_event("agent_complete", {})
    except GeneratorExit:
        log_event("agent_cancelled", {})


@app.function(
    image=orchestrator_image,
    volumes={"/state": graph_state_vol},
)
def clear_graph_state() -> str:
    """One-time cleanup: wipe all persisted graph state. Run once after deploy."""
    import shutil
    from pathlib import Path
    state_dir = Path("/state")
    for f in state_dir.glob("*.json"):
        f.unlink()
    prop_dir = state_dir / "property_graph"
    if prop_dir.exists():
        shutil.rmtree(prop_dir)
    return "State cleared."


@app.function(
    image=orchestrator_image,
    volumes={
        "/state": graph_state_vol,
        "/workspace": workspace_vol,
    },
    timeout=30 * 60,
    retries=0,
)
def run_agent(user_prompt: str) -> dict:
    """Execute a single agent loop iteration for *user_prompt* synchronously.

    Returns the final episodic graph state serialised as JSON-compatible dict.
    """
    from orchestrator.agent_loop import agent_loop

    log_event("agent_start", {"prompt": user_prompt[:200]})

    result = agent_loop(user_prompt)

    log_event("agent_complete", {"nodes": len(result.get("nodes", []))})
    return result


@app.function(
    image=orchestrator_image,
    volumes={"/state": graph_state_vol},
    timeout=60,
)
def read_graph_state() -> dict:
    """Return the current Tri-Graph state as a JSON-serialisable dict.

    Called by the Chainlit UI knowledge-browser action to load persisted
    graphs without triggering a full agent run.  Returns node/edge counts
    and the full node-link serialisation for each graph.
    """
    import json as _json
    from orchestrator.graph_memory import (
        TriGraphMemory,
        SEMANTIC_PATH,
        PROCEDURAL_PATH,
        EPISODIC_PATH,
    )
    import networkx as nx

    result: dict[str, object] = {
        "episodic":   {"nodes": [], "edges": [], "count": 0},
        "semantic":   {"nodes": [], "edges": [], "count": 0},
        "procedural": {"nodes": [], "edges": [], "count": 0},
        "total_nodes": 0,
        "persisted":   False,
    }

    any_exist = any(p.exists() for p in (SEMANTIC_PATH, PROCEDURAL_PATH, EPISODIC_PATH))
    if not any_exist:
        return result

    result["persisted"] = True
    mem = TriGraphMemory.load()

    for graph_name, nx_graph in [
        ("episodic",   mem.episodic),
        ("semantic",   mem.semantic),
        ("procedural", mem.procedural),
    ]:
        nodes = []
        for node_id, data in nx_graph.nodes(data=True):
            nodes.append({
                "id":      node_id,
                "label":   data.get("label", graph_name.capitalize()),
                "content": data.get("content") or data.get("description") or "",
                "status":  data.get("status", ""),
            })
        edges = [
            {"source": u, "target": v, "relation": d.get("relation", "")}
            for u, v, d in nx_graph.edges(data=True)
        ]
        result[graph_name] = {
            "nodes": nodes,
            "edges": edges,
            "count": len(nodes),
        }

    result["total_nodes"] = (
        result["episodic"]["count"]      # type: ignore[index]
        + result["semantic"]["count"]    # type: ignore[index]
        + result["procedural"]["count"]  # type: ignore[index]
    )
    return result


@app.function(
    image=orchestrator_image,
    volumes={"/state": graph_state_vol},
    timeout=15 * 60,
)
def onboard_domain(
    domain_name: str,
    artifacts: list[str],
    artifact_filenames: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Seed knowledge graphs for a domain from provided artifact texts.

    Extraction is static-first via extractor.py. For code artifacts,
    ast (Python) or tree-sitter (JS/TS) produce typed nodes directly.
    The LLM is called only for missing docstrings, ambiguous concept_type
    classifications, and ConstraintNode fields from prose.

    Args:
        domain_name:        Short identifier, e.g. "trading_dsl"
        artifacts:          List of raw text strings.
        artifact_filenames: Optional filenames (same order as artifacts).
                            Used by the extractor to choose the right
                            parser. Defaults to "artifact_N.txt".
        force:              Re-run even if domain was previously onboarded.

    Returns:
        dict: domain, semantic_nodes_added, procedural_nodes_added
    """
    import json as _json
    from pathlib import Path
    from orchestrator.graph_memory import TriGraphMemory
    from orchestrator.extractor import extract_from_artifact
    from orchestrator.node_schemas import SemanticNode, ProceduralNode
    from pydantic import TypeAdapter, ValidationError

    profile_path = Path("/state/domain_profiles") / f"{domain_name}.json"
    if profile_path.exists() and not force:
        log_event("domain_already_onboarded", {"domain": domain_name})
        return _json.loads(profile_path.read_text())

    filenames = artifact_filenames or [
        f"artifact_{i}.txt" for i in range(len(artifacts))
    ]
    memory = TriGraphMemory.load()
    sem_count = proc_count = 0
    sem_adapter = TypeAdapter(SemanticNode)
    proc_adapter = TypeAdapter(ProceduralNode)

    for text, filename in zip(artifacts, filenames):
        nodes = extract_from_artifact(text, filename)
        for node_dict in nodes:
            graph_type = node_dict.get("graph_type", "")
            if not node_dict.get("node_id"):
                continue
            if graph_type == "semantic":
                try:
                    validated = sem_adapter.validate_python(node_dict)
                    if validated.node_id not in memory.semantic:
                        memory.add_semantic_node(
                            validated.node_id, validated.model_dump()
                        )
                        sem_count += 1
                except (ValidationError, Exception):
                    pass
            elif graph_type == "procedural":
                try:
                    validated = proc_adapter.validate_python(node_dict)
                    if validated.node_id not in memory.procedural:
                        memory.add_procedural_node(
                            validated.node_id, validated.model_dump()
                        )
                        proc_count += 1
                except (ValidationError, Exception):
                    pass

    memory.save()

    result = {
        "domain": domain_name,
        "semantic_nodes_added": sem_count,
        "procedural_nodes_added": proc_count,
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(_json.dumps(result, indent=2))
    log_event("domain_onboarded", result)
    return result


@app.function(
    image=orchestrator_image,
    volumes={"/state": graph_state_vol},
    timeout=15 * 60,
)
def onboard_from_urls(
    domain_name: str,
    urls: list[str],
    depth: int = 1,
    same_domain_only: bool = True,
    force: bool = False,
) -> dict:
    """Fetch URLs and index their content into the knowledge graph.

    Crawls seed_urls to the specified depth, extracts typed nodes via
    extractor.py, and inserts them into the semantic and procedural graphs.

    Args:
        domain_name:      Short identifier for the domain being indexed.
        urls:             Seed URLs to fetch.
        depth:            Link-follow depth. 0 = seeds only, 1 = seeds + links.
        same_domain_only: Only follow links on the same domain as seeds.
        force:            Re-index even if domain profile exists.

    Returns:
        dict: domain, pages_fetched, semantic_nodes_added, procedural_nodes_added
    """
    import json as _json
    from pathlib import Path
    from orchestrator.document_fetcher import crawl_urls
    from orchestrator.extractor import extract_from_artifact
    from orchestrator.graph_memory import TriGraphMemory
    from orchestrator.node_schemas import SemanticNode, ProceduralNode
    from pydantic import TypeAdapter, ValidationError

    profile_path = Path("/state/domain_profiles") / f"{domain_name}_urls.json"
    if profile_path.exists() and not force:
        log_event("domain_urls_already_indexed", {"domain": domain_name})
        return _json.loads(profile_path.read_text())

    docs = crawl_urls(urls, depth=depth, same_domain_only=same_domain_only,
                      max_pages=100)
    memory = TriGraphMemory.load()
    sem_adapter = TypeAdapter(SemanticNode)
    proc_adapter = TypeAdapter(ProceduralNode)
    sem_count = proc_count = 0

    for doc in docs:
        if not doc.content or doc.error:
            continue
        node_dicts = extract_from_artifact(
            doc.content,
            filename=doc.url,
            content_hint=doc.content_type,
        )
        for nd in node_dicts:
            graph_type = nd.get("graph_type", "")
            node_id = nd.get("node_id", "")
            if not node_id:
                continue
            if graph_type == "semantic":
                try:
                    v = sem_adapter.validate_python(nd)
                    if v.node_id not in memory.semantic:
                        memory.add_semantic_node(v.node_id, v.model_dump())
                        sem_count += 1
                except (ValidationError, Exception):
                    pass
            elif graph_type == "procedural":
                try:
                    v = proc_adapter.validate_python(nd)
                    if v.node_id not in memory.procedural:
                        memory.add_procedural_node(v.node_id, v.model_dump())
                        proc_count += 1
                except (ValidationError, Exception):
                    pass

    memory.save()
    result = {
        "domain": domain_name,
        "pages_fetched": len(docs),
        "semantic_nodes_added": sem_count,
        "procedural_nodes_added": proc_count,
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(_json.dumps(result, indent=2))
    log_event("domain_urls_indexed", result)
    return result
