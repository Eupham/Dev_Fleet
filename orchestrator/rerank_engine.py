"""Rerank Engine — Scores edge relevance between task nodes and graph candidates.

Uses the 7B model (or a lightweight embedding model) to compute a relevance
score between an ``AtomicTaskNode`` and candidate nodes from the Semantic
Graph.  Edges with score > 0.75 are created.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schema for scored edges
# ---------------------------------------------------------------------------


class ScoredEdge(BaseModel):
    """A scored relationship between an episodic task and a semantic node."""

    task_id: str
    candidate_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Reranking prompt template
# ---------------------------------------------------------------------------

RERANK_SYSTEM = """You are a relevance scoring engine.
Given a TASK description and a CANDIDATE code element, respond with ONLY
a JSON object: {"score": <float 0.0-1.0>, "rationale": "<one sentence>"}.
Score 1.0 = perfectly relevant, 0.0 = completely irrelevant."""


def _build_rerank_messages(
    task_desc: str, candidate_desc: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {
            "role": "user",
            "content": f"TASK: {task_desc}\nCANDIDATE: {candidate_desc}",
        },
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

EDGE_THRESHOLD = 0.75


def rerank_candidates(
    task_id: str,
    task_description: str,
    candidates: list[dict[str, Any]],
    inference_url: str,
    model: str = "llm",
    timeout: float = 60.0,
) -> list[ScoredEdge]:
    """Score each *candidate* against the given task via the LLM.

    Parameters
    ----------
    task_id:
        ID of the ``AtomicTaskNode``.
    task_description:
        Human-readable task description.
    candidates:
        List of dicts with at least ``{"id": str, "description": str}``.
    inference_url:
        Base URL of the vLLM inference service.
    model:
        Served model name.
    timeout:
        Per-request HTTP timeout.

    Returns
    -------
    List of ``ScoredEdge`` objects with score > ``EDGE_THRESHOLD``.
    """
    import json as _json

    edges: list[ScoredEdge] = []

    for cand in candidates:
        messages = _build_rerank_messages(
            task_description, cand.get("description", "")
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
        }

        try:
            resp = httpx.post(
                f"{inference_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            result = _json.loads(content)
            score = float(result.get("score", 0.0))
            rationale = result.get("rationale", "")
        except Exception:
            score = 0.0
            rationale = "scoring_error"

        if score > EDGE_THRESHOLD:
            edges.append(
                ScoredEdge(
                    task_id=task_id,
                    candidate_id=cand["id"],
                    score=score,
                    rationale=rationale,
                )
            )

    return edges
