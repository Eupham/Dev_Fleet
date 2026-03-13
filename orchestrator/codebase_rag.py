"""Codebase RAG — Repository context retrieval node.

Isolates the bi-encoder retrieval step so agent_loop.py stays clean.
Queries the PropertyGraphIndex with the user prompt and returns a
formatted context string injected into the downstream Decompose node.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.agent_loop import AgentState

logger = logging.getLogger("dev_fleet.codebase_rag")


def retrieve_codebase_node(state: "AgentState") -> dict:
    """Retrieve relevant codebase nodes from the Tri-Graph PropertyGraph index.

    Uses the user prompt as the query against the semantic + procedural
    knowledge graph (top-k=3) and formats results into a plain-text context
    block that the Decompose node will inject into the Frege prompt.
    """
    from orchestrator.graph_memory import TriGraphMemory

    logger.info("Retrieve_Codebase node querying PropertyGraph...")

    memory = TriGraphMemory.load()

    try:
        retriever = memory.property_graph.as_retriever(similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(state["user_prompt"])
    except Exception as exc:
        logger.warning("Codebase retrieval failed (%s). Proceeding with empty context.", exc)
        return {
            "codebase_context": "",
            "messages": ["Codebase retrieval failed; proceeding without context."],
        }

    lines: list[str] = []
    for node in retrieved_nodes:
        node_id = node.metadata.get("node_id", "unknown")
        graph_type = node.metadata.get("graph_type", "unknown")
        lines.append(f"[{graph_type}:{node_id}] {node.text.strip()}")

    codebase_context = "\n".join(lines)
    logger.info("Retrieved %d codebase nodes for context.", len(retrieved_nodes))

    return {
        "codebase_context": codebase_context,
        "messages": ["Retrieved codebase context."],
    }
