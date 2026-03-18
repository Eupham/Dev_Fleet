"""Task difficulty scoring via Epistemic (Reranker) and Structural (AST) signals."""
from __future__ import annotations
import ast
import networkx as nx

def ast_kolmogorov_complexity(source_code: str) -> float:
    """Proxy for Kolmogorov complexity via AST density analysis.
    
    Measures structural programmatic length. Higher density (nodes/identifiers) 
    implies higher algorithmic complexity.
    """
    if not source_code or not source_code.strip():
        return 0.0
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return 1.0 # Max difficulty for unparseable code
    
    # Count nodes (Structure) and identifiers (Information)
    node_count = sum(1 for _ in ast.walk(tree))
    identifiers = set(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    
    # Normalized density metric
    return min(1.0, (node_count * len(identifiers)) / 1000.0)

def compute_base_difficulty(
    task_id: str,
    task_description: str,
    reranker_edges: list,
    composition_graph: nx.DiGraph,
    target_code: str = "",
    w_coverage: float = 0.50,    # Weight for Epistemic Knowledge
    w_structure: float = 0.25,   # Weight for Graph Topology
    w_kolmogorov: float = 0.25,  # Weight for Algorithmic Density
) -> float:
    """Compute base difficulty from Epistemic and Structural signals."""
    
    # Signal 1: Reranker Coverage (Epistemic)
    # Measures how well existing knowledge (Semantic/Procedural graphs) covers the task.
    scores = [e.score for e in reranker_edges if e.task_id == task_id]
    coverage_score = sum(scores) / len(scores) if scores else 0.0
    coverage_difficulty = 1.0 - coverage_score

    # Signal 2: Structural Load (DAG)
    declared = nx.DiGraph([(u,v) for u,v,d in composition_graph.edges(data=True) if d.get("edge_type") == "declared"])
    structure = 0.0
    if task_id in declared and nx.is_directed_acyclic_graph(declared):
        in_deg = declared.in_degree(task_id)
        out_deg = declared.out_degree(task_id)
        structure = min(1.0, in_deg * 0.2 + out_deg * 0.1)

    # Signal 3: Kolmogorov Proxy (AST Density)
    kolmogorov = ast_kolmogorov_complexity(target_code)

    return min(1.0, 
        w_coverage * coverage_difficulty + 
        w_structure * structure + 
        w_kolmogorov * kolmogorov
    )

def difficulty_to_tier(score: float) -> str:
    """Map difficulty score to model routing tier."""
    if score < 0.20: return "trivial"
    if score < 0.40: return "simple"
    if score < 0.60: return "moderate"
    if score < 0.80: return "complex"
    return "expert"
