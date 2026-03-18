"""Task difficulty scoring — Kolmogorov (compression-based) + Epistemic + Structural.
FIXES APPLIED:
1. Replaced AST-density proxy with gzip-based Normalized Compression Distance (NCD).
   AST node_count * identifiers is NOT a Kolmogorov approximation — it measures code
   density, not algorithmic complexity. Gzip length is a standard K(x) upper bound.
2. Added conditional_complexity() — measures K(task | knowledge) via compression
   distance, which is the actual quantity we want for difficulty estimation.
3. Fixed magic-number weights with documented justification.
4. Added reassess_remaining_tasks() for post-execution difficulty updates.
"""
from __future__ import annotations
import gzip
import networkx as nx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from orchestrator.composition import CompositionLedger
# ---------------------------------------------------------------------------
# Signal 1: Kolmogorov Proxy via Compression
# ---------------------------------------------------------------------------
def compression_complexity(text: str) -> float:
    """Approximate K(text) via gzip compression ratio.
    Returns a normalized score in [0, 1] where:
      - 0.0 = highly compressible (repetitive, low complexity)
      - 1.0 = incompressible (high algorithmic information content)
    This is a standard approximation: gzip length provides an upper bound
    on Kolmogorov complexity (Li & Vitányi, "An Introduction to Kolmogorov
    Complexity and Its Applications", Chapter 8).
    """
    if not text or len(text) < 10:
        return 0.0
    raw = text.encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9)
    # Ratio: compressed_size / raw_size. Pure random ≈ 1.0, repetitive ≈ 0.0
    return min(1.0, len(compressed) / len(raw))
def conditional_complexity(task_description: str, knowledge_context: str) -> float:
    """Approximate K(task | knowledge) — conditional Kolmogorov complexity.
    Measures how much NEW information the task requires beyond what the
    knowledge graph already provides. Uses Normalized Information Distance:
        NID(x, y) ≈ (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    where C(·) is gzip length. High NID means the task is informationally
    distant from existing knowledge (harder). Low NID means existing
    knowledge already covers it (easier).
    This is the core insight: difficulty should be relative to what we
    already know, not absolute.
    """
    if not task_description:
        return 0.0
    if not knowledge_context or len(knowledge_context) < 10:
        # No knowledge → maximum conditional complexity
        return compression_complexity(task_description)
    def _c(s: str) -> int:
        return len(gzip.compress(s.encode("utf-8"), compresslevel=9))
    c_task = _c(task_description)
    c_know = _c(knowledge_context)
    c_joint = _c(task_description + "\n" + knowledge_context)
    denominator = max(c_task, c_know)
    if denominator == 0:
        return 0.0
    nid = (c_joint - min(c_task, c_know)) / denominator
    return min(1.0, max(0.0, nid))
# ---------------------------------------------------------------------------
# Signal 2: Epistemic Coverage (Reranker)
# ---------------------------------------------------------------------------
def epistemic_coverage(task_id: str, reranker_edges: list) -> float:
    """How well does the knowledge graph cover this task? [0=no coverage, 1=full].

    Uses cross-encoder reranker scores from the Tri-Graph retrieval.
    High coverage -> low difficulty (we already know how to do this).

    Scores are filtered to edges whose task_id matches, or to all edges when
    the reranker returns global (non-task-specific) nodes — the previous
    ``or True`` caused every edge to count for every task, defeating the
    per-task assessment entirely.
    """
    # Prefer task-specific edges; fall back to all edges only when none are
    # tagged with a task_id (i.e. the reranker returned generic knowledge nodes).
    task_specific = [
        e.score for e in reranker_edges
        if getattr(e, "task_id", None) == task_id
    ]
    scores = task_specific if task_specific else [
        e.score for e in reranker_edges if hasattr(e, "score")
    ]
    if not scores:
        return 0.0  # No coverage = maximum difficulty
    return sum(scores) / len(scores)
# ---------------------------------------------------------------------------
# Signal 3: Structural Load (DAG topology)
# ---------------------------------------------------------------------------
def structural_load(task_id: str, composition_graph: nx.DiGraph) -> float:
    """Difficulty contribution from DAG position — fan-in creates integration load.
    Tasks with many predecessors (high in-degree) must integrate multiple
    outputs → harder. Tasks with many successors (high out-degree) are
    foundation tasks but not inherently harder to execute.
    Only counts 'declared' edges (causal dependencies), not 'observed'
    edges (filesystem co-occurrence which can be spurious).
    """
    declared = nx.DiGraph([
        (u, v) for u, v, d in composition_graph.edges(data=True)
        if d.get("edge_type") == "declared"
    ])
    if task_id not in declared or not nx.is_directed_acyclic_graph(declared):
        return 0.0
    in_deg = declared.in_degree(task_id)
    out_deg = declared.out_degree(task_id)
    # Fan-in (integration) weighted 3× more than fan-out (dependency)
    return min(1.0, in_deg * 0.25 + out_deg * 0.05)
# ---------------------------------------------------------------------------
# Composite Difficulty Score
# ---------------------------------------------------------------------------
def compute_base_difficulty(
    task_id: str,
    task_description: str,
    reranker_edges: list,
    composition_graph: nx.DiGraph,
    knowledge_context: str = "",
    # Weight justification:
    # - Conditional K dominates because it directly measures "how much new
    #   information does this task require beyond what we know?"
    # - Epistemic coverage is the reranker's view of the same question
    #   (redundant but from a different signal source)
    # - Structural load is a small penalty for integration complexity
    w_kolmogorov: float = 0.50,
    w_coverage: float = 0.35,
    w_structure: float = 0.15,
) -> float:
    """Compute base difficulty from three orthogonal signals.
    Returns a score in [0, 1] where 0 = trivial, 1 = maximum difficulty.
    """
    # Signal 1: Conditional Kolmogorov — K(task | knowledge)
    k_score = conditional_complexity(task_description, knowledge_context)
    # Signal 2: Epistemic coverage (inverted — high coverage = low difficulty)
    coverage = epistemic_coverage(task_id, reranker_edges)
    coverage_difficulty = 1.0 - coverage
    # Signal 3: Structural load from DAG position
    structure = structural_load(task_id, composition_graph)
    return min(1.0,
        w_kolmogorov * k_score +
        w_coverage * coverage_difficulty +
        w_structure * structure
    )
# ---------------------------------------------------------------------------
# Tier Routing
# ---------------------------------------------------------------------------
# Maps difficulty score to model/GPU tier.
# These tiers must correspond to actual inference endpoints in model_pool.py.
TIER_THRESHOLDS = [
    (0.20, "trivial"),    # Pattern matching, no LLM needed (future: template)
    (0.40, "simple"),     # Small model, fast GPU (e.g., Qwen-0.5B on T4)
    (0.60, "moderate"),   # Medium model (e.g., Qwen-7B on A10G)
    (0.80, "complex"),    # Large model (e.g., Qwen-35B on L40S)
]
def difficulty_to_tier(score: float) -> str:
    """Map difficulty score to model routing tier."""
    for threshold, tier in TIER_THRESHOLDS:
        if score < threshold:
            return tier
    return "expert"  # score >= 0.80 → largest model, best GPU
# ---------------------------------------------------------------------------
# Post-Execution Reassessment
# ---------------------------------------------------------------------------
def reassess_remaining_tasks(
    remaining_tasks: list[dict],
    knowledge_context: str,
    reranker_edges: list,
    composition_graph: nx.DiGraph,
) -> list[tuple[str, float, str]]:
    """Re-score all remaining tasks after a task execution updated the graph.
    Returns [(task_id, new_score, new_tier), ...] for each remaining task.
    This is the critical feedback loop: task[i]'s execution may have:
    - Added knowledge nodes (lowering K(task[i+1] | knowledge))
    - Introduced new dependencies (raising structural load)
    - Discovered unknown libraries (raising K if no knowledge exists)
    Called after build_knowledge_graphs() in the agent loop.
    """
    results = []
    for task in remaining_tasks:
        tid = task.get("id", "")
        desc = task.get("description", "")
        score = compute_base_difficulty(
            task_id=tid,
            task_description=desc,
            reranker_edges=reranker_edges,
            composition_graph=composition_graph,
            knowledge_context=knowledge_context,
        )
        tier = difficulty_to_tier(score)
        results.append((tid, score, tier))
    return results
