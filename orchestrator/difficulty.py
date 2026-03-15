"""Compositional Difficulty Derivation — no LLM, no classification.

Difficulty is derived from two signals the system already computes:
1. Reranker coverage: mean reranker score for a task's knowledge graph matches.
   High score = well-covered by existing knowledge = lower difficulty.
2. Graph topology: position in the composition graph (in-degree, out-degree, depth).

Difficulty propagates through composition edges (Frege application):
a task inherits partial difficulty from its hardest predecessor.
This means the difficulty of the composite is a function of the difficulties
of its parts and their combination rules (the edges).

On cold start with empty knowledge graphs, novelty = 1.0 for all tasks.
Difficulty is driven entirely by graph topology — correct default behaviour,
since with no prior knowledge the system should prefer stronger models.
"""

from __future__ import annotations

import networkx as nx


def compute_base_difficulty(
    task_id: str,
    reranker_edges: list,  # list[ScoredEdge]
    composition_graph: nx.DiGraph,
) -> float:
    """Derive base difficulty from reranker coverage and graph topology.

    No LLM call. No classification. Two measured signals combined.

    Parameters
    ----------
    task_id:
        ID of the task to score.
    reranker_edges:
        ScoredEdge objects from the reranker (with .task_id and .score).
    composition_graph:
        The composition graph (either pre-execution DAG or post-execution
        observed graph from CompositionLedger).

    Returns
    -------
    float in [0.0, 1.0] — higher means harder.
    """
    # Coverage: mean reranker score for this task's knowledge graph matches.
    # High score = well-covered by existing knowledge = lower difficulty.
    scores = [e.score for e in reranker_edges if e.task_id == task_id]
    coverage = sum(scores) / len(scores) if scores else 0.0
    novelty = 1.0 - coverage

    # Structural load from composition graph position.
    if task_id not in composition_graph:
        structure = 0.0
    else:
        in_degree = composition_graph.in_degree(task_id)
        out_degree = composition_graph.out_degree(task_id)

        roots = [n for n in composition_graph.nodes()
                 if composition_graph.in_degree(n) == 0]
        depth = 0
        for root in roots:
            try:
                for path in nx.all_simple_paths(composition_graph, root, task_id):
                    depth = max(depth, len(path) - 1)
            except nx.NetworkXNoPath:
                pass

        structure = min(1.0, in_degree * 0.2 + out_degree * 0.1 + depth * 0.15)

    return min(1.0, novelty * 0.6 + structure * 0.4)


def propagate_difficulty(
    composition_graph: nx.DiGraph,
    base_difficulty: dict[str, float],
    propagation_weight: float = 0.3,
) -> dict[str, float]:
    """Propagate difficulty through composition edges in topological order.

    A task inherits difficulty from its hardest predecessor. This is the
    Frege application: the difficulty of the whole is determined by the
    difficulties of its parts and their combination structure.

    Parameters
    ----------
    composition_graph:
        DAG of task dependencies.
    base_difficulty:
        Per-task base difficulty from compute_base_difficulty.
    propagation_weight:
        How much of the predecessor's max difficulty is inherited (default 0.3).

    Returns
    -------
    dict mapping task_id → propagated difficulty score in [0.0, 1.0].
    """
    final: dict[str, float] = {}

    try:
        order = list(nx.topological_sort(composition_graph))
    except nx.NetworkXUnfeasible:
        # Cycle in graph (shouldn't happen with observed composition, but be safe)
        order = list(composition_graph.nodes())

    for task_id in order:
        predecessors = list(composition_graph.predecessors(task_id))
        base = base_difficulty.get(task_id, 0.5)
        if not predecessors:
            final[task_id] = base
        else:
            inherited = max(final.get(p, 0.0) for p in predecessors)
            final[task_id] = min(1.0, base + inherited * propagation_weight)

    return final


# Thresholds are tunable. The mapping is mechanical — no LLM involved.
def difficulty_to_tier(score: float) -> str:
    """Map a propagated difficulty score to a routing tier.

    Tiers correspond to the model routing table:
      trivial  → Qwen3-4B (T4)
      simple   → Qwen3-8B (T4)
      moderate → Qwen3-Coder-30B-A3B-Instruct (A10G)
      complex  → Qwen3-Coder-30B-A3B-Instruct (A10G)
      expert   → Qwen3-Coder-480B-A35B-Instruct (A100-80GB)
    """
    if score < 0.15:
        return "trivial"
    if score < 0.35:
        return "simple"
    if score < 0.55:
        return "moderate"
    if score < 0.80:
        return "complex"
    return "expert"
