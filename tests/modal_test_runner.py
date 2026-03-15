"""Modal-native tests — run the graph memory, parser, and sandbox on Modal.

Usage:
    modal run tests/modal_test_runner.py

All tests execute inside Modal containers. No local mocking.
"""

from __future__ import annotations

import json
import sys

import modal

app = modal.App("devfleet-tests")

test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "networkx>=3.2",
        "pydantic>=2.5",
        "llama-index-core>=0.10.0",
        "llama-index>=0.10.0",
        "llama-index-embeddings-huggingface>=0.1.0",
    )
    .add_local_dir("orchestrator", remote_path="/root/orchestrator", copy=True)
    .add_local_dir("inference", remote_path="/root/inference", copy=True)
    .add_local_python_source("fleet_app", copy=True)
)


@app.function(image=test_image, timeout=600)
def test_graph_memory() -> str:
    """Test TriGraphMemory: add nodes/edges, save/load, build context."""
    import tempfile
    from pathlib import Path

    sys.path.insert(0, "/root")

    import orchestrator.graph_memory as gm

    # Redirect persistence to a temp dir
    tmp = Path(tempfile.mkdtemp())
    gm.STATE_DIR = tmp
    gm.SEMANTIC_PATH = tmp / "semantic_graph.json"
    gm.PROCEDURAL_PATH = tmp / "procedural_graph.json"
    gm.EPISODIC_PATH = tmp / "episodic_graph.json"

    from orchestrator.graph_memory import TriGraphMemory

    mem = TriGraphMemory()

    # Add nodes
    mem.add_episodic_node("task-1", {"description": "Write tests"})
    assert "task-1" in mem.episodic, "Episodic node not added"

    mem.add_semantic_node("fn_sort", {"type": "function", "name": "sort"})
    assert "fn_sort" in mem.semantic, "Semantic node not added"

    mem.add_procedural_node("rule-1", {"rule": "always lint"})
    assert "rule-1" in mem.procedural, "Procedural node not added"

    # Edges
    mem.add_episodic_edge("task-1", "fn_sort", {"graph": "semantic", "score": 0.9})
    assert mem.episodic.has_edge("task-1", "fn_sort"), "Edge not created"

    # Save + load roundtrip
    mem.save()
    restored = TriGraphMemory.load()
    assert "task-1" in restored.episodic, "Roundtrip failed"
    assert restored.episodic.nodes["task-1"]["description"] == "Write tests"

    # Context building
    ctx = mem.build_context("task-1")
    assert "[Episodic]" in ctx, "Missing episodic context"
    assert "[Semantic]" in ctx, "Missing semantic context"

    ctx_empty = mem.build_context("nonexistent")
    assert ctx_empty == "(no context)", "Empty context wrong"

    # to_dict
    d = mem.to_dict()
    assert "semantic" in d and "procedural" in d and "episodic" in d

    return "✓ test_graph_memory passed (7 assertions)"


@app.function(image=test_image, timeout=300)
def test_frege_parser_schemas() -> str:
    """Test Pydantic schemas for AtomicTaskNode and TaskDAG."""
    sys.path.insert(0, "/root")
    from orchestrator.task_parser import AtomicTaskNode, TaskDAG, TransformTask

    # Default fields
    node = TransformTask(description="Write a function")
    assert node.status == "pending"
    assert len(node.id) == 12

    # Custom fields
    node2 = TransformTask(
        id="abc123",
        description="Run tests",
        tool_hint="bash",
        status="running",
    )
    assert node2.tool_hint == "bash"

    # Full DAG
    dag = TaskDAG(
        user_prompt="Sort a list",
        tasks=[
            TransformTask(description="Implement merge sort"),
            TransformTask(description="Write unit tests"),
        ],
    )
    assert len(dag.tasks) == 2

    return "✓ test_frege_parser_schemas passed (4 assertions)"


@app.function(image=test_image, timeout=120)
def test_rerank_schemas() -> str:
    """Test Pydantic ScoredEdge schema and bounds checking."""
    sys.path.insert(0, "/root")
    from orchestrator.rerank_engine import ScoredEdge

    edge = ScoredEdge(task_id="t1", candidate_id="c1", score=0.85, rationale="ok")
    assert edge.score == 0.85

    try:
        ScoredEdge(task_id="t1", candidate_id="c1", score=1.5)
        return "✗ Score > 1.0 should have raised"
    except Exception:
        pass

    return "✓ test_rerank_schemas passed (2 assertions)"


@app.function(image=test_image, timeout=300)
def test_sandbox_result() -> str:
    """Test SandboxResult dataclass."""
    sys.path.insert(0, "/root")
    from orchestrator.tool_sandbox import SandboxResult

    ok = SandboxResult(stdout="hello\n", stderr="", exit_code=0)
    assert ok.success is True

    fail = SandboxResult(stdout="", stderr="error", exit_code=1)
    assert fail.success is False

    return "✓ test_sandbox_result passed (2 assertions)"


@app.local_entrypoint()
def main():
    """Run all tests on Modal and print results."""
    print("=" * 60)
    print("Running Dev Fleet tests on Modal")
    print("=" * 60)

    # Launch all tests in parallel
    results = []

    for future in [
        test_graph_memory.spawn(),
        test_frege_parser_schemas.spawn(),
        test_rerank_schemas.spawn(),
        test_sandbox_result.spawn(),
    ]:
        results.append(future.get())

    all_passed = True
    for r in results:
        print(r)
        if not r.startswith("✓"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print(f"All {len(results)} test suites passed on Modal ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
