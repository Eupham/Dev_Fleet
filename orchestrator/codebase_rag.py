"""Codebase RAG — The Cartographer.

Builds a lightweight "Mini-Map" of the repository for the Explorer (Frege
decomposition engine).  Rather than dumping raw file text into the prompt,
we retrieve the most relevant nodes from the PropertyGraphIndex, apply a
relevance score threshold, and emit a compact list of file paths with a
short content hint.  The Explorer uses this map to plan which files it must
read first, before attempting any modifications.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("dev_fleet.codebase_rag")

SCORE_THRESHOLD = 0.70


def retrieve_codebase_node(state: dict) -> dict:
    """Query the Tri-Graph PropertyGraph and return a scored Mini-Map.

    Retrieval pipeline
    ------------------
    1. Bi-encoder ANN: top-10 candidate nodes via PropertyGraphIndex.
    2. Score threshold: discard any node whose relevance score is below 0.70.
    3. Mini-Map formatting: emit only file_path + first 100 chars of text,
       keeping the prompt token budget minimal.

    Returns
    -------
    ``{"codebase_context": mini_map_string}``
        A newline-joined list of ``- <file_path>: (Contains: <hint>...)``
        entries, or an empty string if nothing passes the threshold.
    """
    from orchestrator.graph_memory import TriGraphMemory

    logger.info("Retrieve_Codebase (Cartographer) querying PropertyGraph (top-k=10)...")

    memory = TriGraphMemory.load()

    try:
        retriever = memory.property_graph.as_retriever(similarity_top_k=10)
        retrieved_nodes = retriever.retrieve(state["user_prompt"])
    except Exception as exc:
        logger.warning("Codebase retrieval failed (%s). Proceeding with empty Mini-Map.", exc)
        return {"codebase_context": ""}

    mini_map_lines: list[str] = []
    kept = 0
    for node in retrieved_nodes:
        score = node.score if node.score is not None else 0.0
        if score < SCORE_THRESHOLD:
            continue

        file_path = node.metadata.get("file_path") or node.metadata.get("node_id", "unknown")
        first_line = node.text.strip()[:100].replace("\n", " ")
        mini_map_lines.append(f"- {file_path}: (Contains: {first_line}...)")
        kept += 1

    logger.info(
        "Cartographer: %d/%d nodes passed score threshold %.2f.",
        kept, len(retrieved_nodes), SCORE_THRESHOLD,
    )

    return {"codebase_context": "\n".join(mini_map_lines)}
