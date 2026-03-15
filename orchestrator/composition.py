"""Execution-Grounded Composition — observe filesystem changes, don't declare them.

A task is a function from workspace state to workspace state. Rather than asking
the model to declare its I/O contracts (which it does poorly), we observe them by
diffing the sandbox filesystem before and after each task execution.

The composition type system is dynamic: each task is typed by its observed state
transition. Composition is verified by execution. This is language-agnostic —
it only observes the filesystem, which is the universal interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from orchestrator.tool_sandbox import ModalSandboxTool


@dataclass(frozen=True)
class WorkspaceState:
    """Snapshot of the /workspace filesystem at a point in time."""

    files: frozenset
    file_hashes: dict

    @staticmethod
    def capture(tool: "ModalSandboxTool", timeout: int = 30) -> "WorkspaceState":
        """Capture the current workspace filesystem state via a sandbox call.

        Runs sha256sum on all files in /workspace and returns an immutable
        snapshot of the file tree and content hashes.
        """
        result = tool.forward(
            code="find /workspace -type f -exec sha256sum {} + 2>/dev/null || true",
            language="bash",
            timeout=timeout,
        )
        files = set()
        hashes = {}
        for line in result.get("stdout", "").strip().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                digest, path = parts[0], parts[1]
                hashes[path] = digest
                files.add(path)
        return WorkspaceState(files=frozenset(files), file_hashes=hashes)

    @staticmethod
    def empty() -> "WorkspaceState":
        """Return an empty workspace state (before any task runs)."""
        return WorkspaceState(files=frozenset(), file_hashes={})

    def diff(self, prior: "WorkspaceState") -> "StateDelta":
        """Compute the delta from *prior* to *self*.

        Returns a StateDelta describing which files were created, deleted,
        or modified during the transition from prior to this state.
        """
        created = self.files - prior.files
        deleted = prior.files - self.files
        modified = frozenset(
            p for p in self.files & prior.files
            if self.file_hashes.get(p) != prior.file_hashes.get(p)
        )
        return StateDelta(created=created, deleted=deleted, modified=modified)


@dataclass(frozen=True)
class StateDelta:
    """The filesystem change caused by a single task execution."""

    created: frozenset
    deleted: frozenset
    modified: frozenset

    @property
    def reads(self) -> frozenset:
        """Files that were read (modified or deleted implies prior existence)."""
        return self.modified | self.deleted

    @property
    def writes(self) -> frozenset:
        """Files that were written (created or modified)."""
        return self.created | self.modified

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "created": sorted(self.created),
            "deleted": sorted(self.deleted),
            "modified": sorted(self.modified),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateDelta":
        """Deserialize from a dict."""
        return cls(
            created=frozenset(data.get("created", [])),
            deleted=frozenset(data.get("deleted", [])),
            modified=frozenset(data.get("modified", [])),
        )


class CompositionLedger:
    """Accumulates per-task filesystem deltas and derives the dependency graph.

    Usage:
        ledger = CompositionLedger()
        before = WorkspaceState.capture(tool)
        # ... run task ...
        after = WorkspaceState.capture(tool)
        ledger.record(task_id, before, after)
        ...
        graph = ledger.derive_dependency_graph()
    """

    def __init__(self):
        self.deltas: dict[str, StateDelta] = {}
        # Preserve insertion order for topological reasoning
        self._order: list[str] = []

    def record(self, task_id: str, before: WorkspaceState, after: WorkspaceState) -> None:
        """Record the filesystem delta for a completed task."""
        delta = after.diff(before)
        self.deltas[task_id] = delta
        if task_id not in self._order:
            self._order.append(task_id)

    def derive_dependency_graph(self) -> nx.DiGraph:
        """Derive the dependency graph from observed filesystem transitions.

        An edge a → b exists if task a wrote a file that task b read.
        This is the Frege application: composition structure is derived from
        the observed behavior of parts, not from declared contracts.
        """
        G = nx.DiGraph()
        tasks = list(self.deltas.keys())
        for tid in tasks:
            G.add_node(tid)
        for a in tasks:
            for b in tasks:
                if a != b and self.deltas[a].writes & self.deltas[b].reads:
                    G.add_edge(a, b)

        # Stamp all inferred edges as observed (filesystem co-occurrence)
        for u, v in G.edges():
            G[u][v]["edge_type"] = "observed"

        # Circular filesystem dependencies are logically incoherent for
        # sequential execution. Raise rather than silently return a bad graph.
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError(
                "Derived composition graph is cyclic. Tasks with circular "
                "filesystem dependencies cannot execute sequentially."
            )

        return G

    def to_dict(self) -> dict:
        """Serialize all deltas to JSON-compatible dict."""
        return {tid: delta.to_dict() for tid, delta in self.deltas.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "CompositionLedger":
        """Deserialize from a dict (e.g. from AgentState)."""
        ledger = cls()
        for task_id, delta_dict in data.items():
            ledger.deltas[task_id] = StateDelta.from_dict(delta_dict)
            ledger._order.append(task_id)
        return ledger


def merge_declared_edges(G: nx.DiGraph, tasks: list) -> nx.DiGraph:
    """Stamp declared dependency edges from task preconditions onto G.

    Declared edges reflect actual causal dependencies: task A produces
    output that task B requires. Observed edges (from CompositionLedger)
    reflect filesystem co-occurrence which can be spurious.

    Only declared edges are used for structural difficulty computation.
    Both types are preserved in G for inspection.
    """
    id_set = {t.get("id") for t in tasks}
    for task in tasks:
        src_id = task.get("id")
        for pre_id in task.get("preconditions", []):
            if pre_id in id_set and src_id:
                if G.has_node(pre_id) and G.has_node(src_id):
                    if G.has_edge(pre_id, src_id):
                        G[pre_id][src_id]["edge_type"] = "declared"
                    else:
                        G.add_edge(pre_id, src_id, edge_type="declared")
    return G
