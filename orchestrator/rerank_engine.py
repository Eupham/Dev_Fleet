"""Rerank Engine — Qwen3-Reranker cross-encoder edge scoring.

Uses Qwen3-Reranker-0.6B to assess the relevance between task nodes
(from the Frege-decomposed DAG) and candidate nodes from the three
Knowledge Graphs (Semantic, Procedural, Episodic).

Frege's compositionality principle is applied: scored edges encode
contextual relevance, letting the agent loop derive task complexity
and difficulty from the graph topology.

Edges with score > ``EDGE_THRESHOLD`` are created.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("dev_fleet.rerank")

# ---------------------------------------------------------------------------
# Pydantic schema for scored edges
# ---------------------------------------------------------------------------


class ScoredEdge(BaseModel):
    """A scored relationship between an episodic task and a graph node."""

    task_id: str
    candidate_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

EDGE_THRESHOLD = 0.75


def rerank_candidates(
    task_id: str,
    task_description: str,
    candidates: list[dict[str, Any]],
) -> list[ScoredEdge]:
    """Score each *candidate* against the task using Qwen3-Reranker.

    Parameters
    ----------
    task_id:
        ID of the ``AtomicTaskNode``.
    task_description:
        Human-readable task description.
    candidates:
        List of dicts with at least ``{"id": str, "description": str}``.

    Returns
    -------
    List of ``ScoredEdge`` objects with score > ``EDGE_THRESHOLD``.
    """
    if not candidates:
        return []

    from inference.reranker import Reranker

    cand_descs = [c.get("description", "") for c in candidates]

    try:
        scores = Reranker().score_pairs.remote(task_description, cand_descs)
    except Exception:
        logger.exception("Reranker call failed for task %s", task_id)
        return []

    edges: list[ScoredEdge] = []
    for cand, score in zip(candidates, scores):
        if score > EDGE_THRESHOLD:
            edges.append(
                ScoredEdge(
                    task_id=task_id,
                    candidate_id=cand["id"],
                    score=score,
                    rationale=f"reranker_score={score:.3f}",
                )
            )
    return edges
