# orchestrator/phase_priors.py
"""Phase-aware difficulty priors for cold and warm start.

Three execution phases:
  Phase 0 (cold):  0 tasks completed. No reranker signal. No graph data.
  Phase 1 (warm):  1-19 tasks. Reranker has some signal. Graphs partially filled.
  Phase 2 (fitted): >= 20 tasks. Reranker and compression signals are primary.

In Phase 0 and Phase 1, difficulty is estimated from task type and
execution frame fields. These priors reflect the structural properties
of the task types:
  query     — read-only, no side effects, lowest structural complexity
  transform — writes state, structurally moderate
  verify    — depends on a transform having succeeded, structurally higher
  compose   — sequences multiple transforms, structurally highest

actor_capability and implementation_depth are included because they
are always available from the typed task schema (not derived from graphs).

In Phase 2, compute_base_difficulty() in difficulty.py uses reranker
coverage and compression. The priors here are the Phase 0/1 fallback.
"""
from __future__ import annotations

MIN_TASKS_PHASE_2 = 20

# Difficulty prior per task type.
# Ordering: query < transform < verify < compose
# Based on structural complexity of the type, not empirical data.
TASK_TYPE_PRIOR: dict[str, float] = {
    "query":     0.20,
    "transform": 0.45,
    "verify":    0.50,
    "compose":   0.65,
}

# Additive adjustment for implementation_depth.
# syscall tasks require lower-level system knowledge than library calls.
DEPTH_ADJUSTMENT: dict[str, float] = {
    "library":   0.00,
    "algorithm": 0.10,
    "syscall":   0.20,
}

# Additive adjustment for actor_capability.
# llm_only tasks have no deterministic execution path — harder to verify.
ACTOR_ADJUSTMENT: dict[str, float] = {
    "bash":     0.00,
    "python":   0.00,
    "llm_only": 0.15,
}


def execution_phase(num_completed_tasks: int) -> int:
    """Return 0, 1, or 2 based on task completion count."""
    if num_completed_tasks == 0:
        return 0
    if num_completed_tasks < MIN_TASKS_PHASE_2:
        return 1
    return 2


def prior_difficulty(
    task_type: str = "transform",
    implementation_depth: str = "library",
    actor_capability: str = "python",
    compression: float = 0.0,
) -> float:
    """Estimate difficulty in Phase 0 and Phase 1.

    Weights:
      75% structural prior (task type, implementation_depth, actor_capability —
          depth_adj and actor_adj are folded into typed_prior before this call)
      25% compression signal (always available from description text)
    """
    base = TASK_TYPE_PRIOR.get(task_type, 0.45)
    depth_adj = DEPTH_ADJUSTMENT.get(implementation_depth, 0.0)
    actor_adj = ACTOR_ADJUSTMENT.get(actor_capability, 0.0)
    typed_prior = min(1.0, base + depth_adj + actor_adj)
    # 75% structural prior (task type + depth + actor, all folded into typed_prior)
    # 25% compression signal
    return min(1.0, 0.75 * typed_prior + 0.25 * compression)


def prior_tier(task_difficulty: float, current_tier: str = "moderate") -> str:
    """Escalate one tier on failure in Phase 0 and Phase 1.

    Does not use a causal model. Escalates one step from the current
    tier, or jumps to expert when difficulty > 0.85. This is the correct
    fallback: it is honest about what it knows and does not overfit
    to insufficient data.
    """
    tiers = ["trivial", "simple", "moderate", "complex", "expert"]
    if task_difficulty > 0.85:
        return "expert"
    try:
        current_idx = tiers.index(current_tier)
    except ValueError:
        current_idx = 2
    return tiers[min(current_idx + 1, len(tiers) - 1)]


def seed_graph_from_task(task_dict: dict, memory) -> None:
    """Seed the procedural graph from a Phase 0 task after successful execution.

    In Phase 0 the knowledge graphs are empty, so the reranker has nothing
    to score against for subsequent tasks. This creates a minimal procedural
    node from the task's typed fields so subsequent tasks have some retrieval
    surface.

    Called once per successfully executed task when execution_phase == 0.
    """
    from orchestrator.node_schemas import ExecutionNode
    from pydantic import ValidationError

    node_id = f"bootstrap_{task_dict.get('id', 'unknown')}"
    try:
        node = ExecutionNode(
            node_id=node_id,
            content=task_dict.get("description", "")[:200],
            label="bootstrap",
            graph_type="procedural",
            actor_capability=task_dict.get("actor_capability", "python"),
            implementation_depth=task_dict.get("implementation_depth", "library"),
            execution_cost=task_dict.get("execution_cost", "moderate"),
        )
        if node_id not in memory.procedural:
            memory.add_procedural_node(node_id, node.model_dump())
    except ValidationError:
        pass
