# orchestrator/web_research.py
"""Web research node — fetches URLs and indexes content into knowledge graphs.

Replaces the LLM-script-generation pattern. Takes a list of URLs derived
from the user prompt (extracted by the supervisor or a lightweight LLM call),
fetches them via document_fetcher, extracts typed nodes via extractor, and
inserts them into the knowledge graph. The graph persists — research is
cumulative, not ephemeral.

The codebase_context returned is a compact summary of what was indexed,
not a raw text dump.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.agent_loop import AgentState

logger = logging.getLogger("dev_fleet.web_research")


def _extract_urls_from_prompt(prompt: str) -> list[str]:
    """Extract explicit URLs from the user prompt.

    For prompts that contain URLs directly (e.g. "summarize https://...")
    these are extracted by regex. If no URLs are present, a lightweight
    LLM call generates search-friendly URLs from the prompt topic.
    """
    import re
    explicit = re.findall(r"https?://[^\s\"'<>]+", prompt)
    if explicit:
        return explicit[:10]

    # No explicit URLs — ask LLM for 3 relevant authoritative URLs
    try:
        from orchestrator.llm_client import chat_completion
        result = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly 3 URLs of authoritative documentation or "
                        "specification pages relevant to this topic. "
                        "One URL per line. Only URLs, nothing else."
                    ),
                },
                {"role": "user", "content": prompt[:500]},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        lines = str(result).strip().splitlines()
        return [l.strip() for l in lines if l.strip().startswith("http")][:3]
    except Exception:
        return []


def research_node(state: "AgentState") -> dict:
    """Fetch URLs, extract typed nodes, index into knowledge graph.

    Returns codebase_context as a summary of indexed content.
    Research results are permanent — nodes survive across sessions.
    """
    from orchestrator.document_fetcher import crawl_urls
    from orchestrator.extractor import extract_from_artifact
    from orchestrator.graph_memory import TriGraphMemory
    from orchestrator.node_schemas import SemanticNode, ProceduralNode
    from pydantic import TypeAdapter, ValidationError

    prompt = state["user_prompt"]
    urls = _extract_urls_from_prompt(prompt)

    if not urls:
        logger.info("Research node: no URLs found for prompt.")
        return {
            "codebase_context": "[Research: no URLs identified for this prompt]",
            "messages": ["[RESEARCH] No URLs to fetch."],
        }

    logger.info("Research node: fetching %d URLs.", len(urls))
    docs = crawl_urls(urls, depth=1, same_domain_only=True, max_pages=20)

    memory = TriGraphMemory.load()
    sem_adapter = TypeAdapter(SemanticNode)
    proc_adapter = TypeAdapter(ProceduralNode)
    sem_count = proc_count = 0
    indexed_titles: list[str] = []

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

        if doc.title:
            indexed_titles.append(doc.title)

    memory.save()

    summary_lines = [
        f"Indexed {sem_count} semantic nodes, {proc_count} procedural nodes",
        f"from {len(docs)} pages:",
    ] + [f"  - {t}" for t in indexed_titles[:10]]
    summary = "\n".join(summary_lines)

    return {
        "codebase_context": summary,
        "messages": [f"[RESEARCH] {summary.splitlines()[0]}"],
    }
