# orchestrator/difficulty.py
"""Task difficulty scoring.

Three signals, each independent and always well-defined:

  1. reranker_coverage  — 1 - mean_reranker_score for this task's matches.
     Measures how well existing knowledge covers the task.
     0.0 when graphs empty (Phase 0/1 fallback: phase_priors.prior_difficulty).

  2. compression_ratio  — 1 - (compressed_len / raw_len) via zlib level 9.
     Information-dense descriptions compress poorly → higher difficulty.
     Always available. Proxy for description complexity, not task complexity.
     See Cilibrasi & Vitanyi (2005) for the NCD foundation — zlib is the
     practical approximation, not the theoretical quantity.

  3. cyclomatic_complexity — max cyclomatic complexity of target source code
     via radon. Only meaningful for modification tasks where target_code is
     the code being changed. Optional signal: returns 0.0 when not provided.

propagate_difficulty uses topological_sort — O(n). The prior implementation
used all_simple_paths for depth computation which is O(n!) in path count.
"""
from __future__ import annotations
import zlib
import networkx as nx


def compression_ratio(text: str) -> float:
    """zlib compression ratio as a description complexity proxy.

    Returns the fraction of information that cannot be compressed away.
    Short, repetitive text → low ratio. Dense, varied text → high ratio.
    Range: [0.0, 1.0].
    """
    encoded = text.encode("utf-8")
    if len(encoded) < 10:
        return 0.0
    compressed_len = len(zlib.compress(encoded, level=9))
    return 1.0 - (compressed_len / len(encoded))


def cyclomatic_complexity_score(source_code: str) -> float:
    """Max cyclomatic complexity of source_code, normalised to [0, 1].

    Returns 0.0 if radon is not installed, source_code is empty, or
    the code is not valid Python. This signal is supplementary —
    the system operates correctly when it returns 0.0.
    """
    if not source_code:
        return 0.0
    try:
        from radon.complexity import cc_visit
        blocks = cc_visit(source_code)
        if not blocks:
            return 0.0
        return min(1.0, max(b.complexity for b in blocks) / 25.0)
    except Exception:
        return 0.0


def compute_base_difficulty(
    task_id: str,
    task_description: str,
    reranker_edges: list,
    composition_graph: nx.DiGraph,
    target_code: str = "",
    w_coverage: float = 0.50,
    w_structure: float = 0.25,
    w_compression: float = 0.15,
    w_code: float = 0.10,
) -> float:
    """Compute base difficulty from up to three signals.

    Uses declared edges only for structural load (edge_type="declared").
    Observed edges from filesystem co-occurrence are excluded — they may
    be spurious and should not inflate structural difficulty.

    Weights are hyperparameters. The defaults favour reranker coverage
    because it is the most task-specific signal when graphs are populated.
    """
    # Signal 1: reranker coverage
    scores = [e.score for e in reranker_edges if e.task_id == task_id]
    coverage_score = sum(scores) / len(scores) if scores else 0.0
    coverage_difficulty = 1.0 - coverage_score

    # Signal 2: structural load from declared dependency edges only
    declared = nx.DiGraph([
        (u, v) for u, v, d in composition_graph.edges(data=True)
        if d.get("edge_type") == "declared"
    ])
    structure = 0.0
    if task_id in declared and nx.is_directed_acyclic_graph(declared):
        in_deg = declared.in_degree(task_id)
        out_deg = declared.out_degree(task_id)
        # O(V+E) depth via topological DP — not all_simple_paths
        depths = {n: 0 for n in declared.nodes()}
        for n in nx.topological_sort(declared):
            for successor in declared.successors(n):
                depths[successor] = max(depths[successor], depths[n] + 1)
        depth = depths.get(task_id, 0)
        structure = min(1.0, in_deg * 0.2 + out_deg * 0.1 + depth * 0.15)

    # Signal 3: compression ratio (always available)
    compress = compression_ratio(task_description)

    # Signal 4: cyclomatic complexity (modification tasks only)
    code_signal = cyclomatic_complexity_score(target_code) if target_code else 0.0

    return min(1.0,
        w_coverage * coverage_difficulty
        + w_structure * structure
        + w_compression * compress
        + w_code * code_signal
    )


def propagate_difficulty(
    composition_graph: nx.DiGraph,
    base_difficulty: dict[str, float],
    propagation_weight: float = 0.3,
) -> dict[str, float]:
    """Propagate difficulty through composition edges. O(n) topological sort.

    A task's final difficulty is its base difficulty plus a fraction of
    its hardest predecessor's difficulty. This reflects that harder
    upstream work tends to produce harder downstream dependencies.
    """
    if not nx.is_directed_acyclic_graph(composition_graph):
        raise ValueError(
            "composition_graph must be a DAG for topological propagation."
        )
    propagated = dict(base_difficulty)
    for node in nx.topological_sort(composition_graph):
        preds = list(composition_graph.predecessors(node))
        if preds:
            max_pred = max(propagated.get(p, 0.0) for p in preds)
            propagated[node] = min(
                1.0,
                propagated.get(node, 0.0) + propagation_weight * max_pred,
            )
    return propagated


def difficulty_to_tier(score: float) -> str:
    """Map difficulty score to model routing tier."""
    if score < 0.20:
        return "trivial"
    if score < 0.40:
        return "simple"
    if score < 0.60:
        return "moderate"
    if score < 0.80:
        return "complex"
    return "expert"
