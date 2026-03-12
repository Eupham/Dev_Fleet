"""Tri-Graph Memory — Semantic, Procedural, and Episodic graphs.

Each graph is a ``networkx.DiGraph``.  Utility helpers merge the current
Episodic node with relevant Semantic/Procedural neighbours into a
localized text representation (GraphRAG context) that is sent to the 32B
model for generation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# Persistence paths (inside the mounted Modal Volume at /state)
# ---------------------------------------------------------------------------

STATE_DIR = Path(os.getenv("DEVFLEET_STATE_DIR", "/state"))

SEMANTIC_PATH = STATE_DIR / "semantic_graph.json"
PROCEDURAL_PATH = STATE_DIR / "procedural_graph.json"
EPISODIC_PATH = STATE_DIR / "episodic_graph.json"


# ---------------------------------------------------------------------------
# Graph wrapper
# ---------------------------------------------------------------------------


@dataclass
class TriGraphMemory:
    """Container for three distinct knowledge graphs."""

    semantic: nx.DiGraph = field(default_factory=nx.DiGraph)
    procedural: nx.DiGraph = field(default_factory=nx.DiGraph)
    episodic: nx.DiGraph = field(default_factory=nx.DiGraph)

    # -- Serialization -------------------------------------------------------

    def save(self) -> None:
        """Persist all three graphs as JSON node-link data."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        for graph, path in [
            (self.semantic, SEMANTIC_PATH),
            (self.procedural, PROCEDURAL_PATH),
            (self.episodic, EPISODIC_PATH),
        ]:
            data = nx.node_link_data(graph)
            path.write_text(json.dumps(data, default=str))

    @classmethod
    def load(cls) -> "TriGraphMemory":
        """Restore graphs from disk, or create empty ones if absent."""
        mem = cls()
        for attr, path in [
            ("semantic", SEMANTIC_PATH),
            ("procedural", PROCEDURAL_PATH),
            ("episodic", EPISODIC_PATH),
        ]:
            if path.exists():
                data = json.loads(path.read_text())
                setattr(mem, attr, nx.node_link_graph(data))
        return mem

    # -- Episodic helpers ----------------------------------------------------

    def add_episodic_node(
        self, node_id: str, attrs: dict[str, Any]
    ) -> None:
        """Add a node to the episodic graph with the given attributes."""
        self.episodic.add_node(node_id, **attrs)

    def add_episodic_edge(
        self, src: str, dst: str, attrs: dict[str, Any] | None = None
    ) -> None:
        """Add a directed edge in the episodic graph."""
        self.episodic.add_edge(src, dst, **(attrs or {}))

    # -- Semantic helpers ----------------------------------------------------

    def add_semantic_node(
        self, node_id: str, attrs: dict[str, Any]
    ) -> None:
        self.semantic.add_node(node_id, **attrs)

    # -- Procedural helpers --------------------------------------------------

    def add_procedural_node(
        self, node_id: str, attrs: dict[str, Any]
    ) -> None:
        self.procedural.add_node(node_id, **attrs)

    # -- GraphRAG context builder --------------------------------------------

    def build_context(self, episodic_node_id: str) -> str:
        """Build a localized text context for *episodic_node_id*.

        Merges the node's own attributes with reachable Semantic and
        Procedural neighbours (1-hop) to form the context window sent to
        the 32B model.
        """
        parts: list[str] = []

        # Episodic node itself
        if episodic_node_id in self.episodic:
            attrs = dict(self.episodic.nodes[episodic_node_id])
            parts.append(f"[Episodic] {episodic_node_id}: {json.dumps(attrs, default=str)}")

        # Linked semantic nodes (via cross-graph edges stored as attrs)
        for _, target, data in self.episodic.out_edges(episodic_node_id, data=True):
            if data.get("graph") == "semantic" and target in self.semantic:
                attrs = dict(self.semantic.nodes[target])
                parts.append(f"[Semantic] {target}: {json.dumps(attrs, default=str)}")

        # Linked procedural nodes
        for _, target, data in self.episodic.out_edges(episodic_node_id, data=True):
            if data.get("graph") == "procedural" and target in self.procedural:
                attrs = dict(self.procedural.nodes[target])
                parts.append(f"[Procedural] {target}: {json.dumps(attrs, default=str)}")

        return "\n".join(parts) if parts else "(no context)"

    def to_dict(self) -> dict[str, Any]:
        """Export all three graphs as a JSON-serializable dictionary."""
        return {
            "semantic": nx.node_link_data(self.semantic),
            "procedural": nx.node_link_data(self.procedural),
            "episodic": nx.node_link_data(self.episodic),
        }
