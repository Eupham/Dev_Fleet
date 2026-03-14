"""Node Labeler — assigns typed labels and content summaries to KG nodes.

Uses the existing vLLM endpoint (Qwen2.5-Coder) with structured Pydantic output.
Qwen does not publish a dedicated labeling/NER model; the instruct variant with
JSON-mode structured output is the idiomatic approach for this task.

For text-document-based KG construction, LlamaIndex's DynamicLLMPathExtractor
(wired to ModalVLLM) is the right tool.  This module covers the programmatic
side: nodes constructed directly from task decomposition results and codebase
RAG snippets that need a typed label + a one-sentence content field.
"""

from __future__ import annotations

import json
from typing import List, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Label taxonomies (per graph type)
# ---------------------------------------------------------------------------

SEMANTIC_LABELS: List[str] = [
    "Concept",
    "Entity",
    "Fact",
    "Principle",
    "Pattern",
    "Definition",
]

PROCEDURAL_LABELS: List[str] = [
    "Tool",
    "Function",
    "Step",
    "Rule",
    "Workflow",
    "Template",
]

# Episodic nodes are always Tasks — no LLM call needed.
EPISODIC_LABEL = "Task"


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------


class NodeClassification(BaseModel):
    label: str = Field(
        description=(
            "Semantic type label for this knowledge node.  Must be one of the "
            "allowed labels supplied in the system prompt."
        )
    )
    content: str = Field(
        description=(
            "One-sentence description of what this node represents, written in "
            "plain English.  Capture the key fact, rule, or capability."
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def label_node(node_id: str, attrs: dict, graph_type: str) -> dict:
    """Classify *node_id* and return ``{"label": str, "content": str}``.

    Calls the vLLM endpoint with structured-output mode (Pydantic schema).
    If labeling fails for any reason the function returns a deterministic
    fallback so downstream code never has to handle errors.

    Parameters
    ----------
    node_id:
        The node's identifier string.
    attrs:
        The node's attribute dict as stored in the NetworkX graph.
    graph_type:
        ``"semantic"``, ``"procedural"``, or ``"episodic"``.

    Returns
    -------
    dict with keys ``label`` (str) and ``content`` (str).
    """
    # Episodic nodes are always Tasks — skip LLM call.
    if graph_type == "episodic":
        return {
            "label": EPISODIC_LABEL,
            "content": str(attrs.get("description", node_id))[:300],
        }

    # Skip if already labeled (avoids re-labeling on _rebuild_property_graph).
    if attrs.get("label") and attrs.get("content"):
        return {"label": attrs["label"], "content": attrs["content"]}

    allowed = SEMANTIC_LABELS if graph_type == "semantic" else PROCEDURAL_LABELS

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a knowledge graph curator. "
                f"Classify this {graph_type} graph node and write a one-sentence content summary. "
                f"The label MUST be one of: {', '.join(allowed)}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"node_id: {node_id}\n"
                f"attributes: {json.dumps(attrs, default=str)[:400]}"
            ),
        },
    ]

    try:
        from orchestrator.llm_client import chat_completion

        result: NodeClassification = chat_completion(
            messages,
            temperature=0.0,
            max_tokens=128,
            schema=NodeClassification,
        )
        # Validate label is in allowed set; fall back to first option if not.
        label = result.label if result.label in allowed else allowed[0]
        return {"label": label, "content": result.content}
    except Exception:
        fallback_label = allowed[0]
        content = str(attrs.get("description") or attrs.get("name") or node_id)[:200]
        return {"label": fallback_label, "content": content}
