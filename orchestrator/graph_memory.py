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
from typing import Any, List

import networkx as nx

from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core import Document, Settings
from llama_index.core.embeddings import BaseEmbedding


# ---------------------------------------------------------------------------
# Zero-Trust Custom Embeddings using Modal RPC
# ---------------------------------------------------------------------------

class ModalEmbeddings(BaseEmbedding):
    """Routes embedding requests to the isolated Modal Embedder service."""

    def _get_query_embedding(self, query: str) -> List[float]:
        from inference.embedder import Embedder
        import modal
        try:
            return Embedder().encode.remote([query])[0]
        except Exception:
            return modal.Cls.from_name("dev_fleet", "Embedder")().encode.remote([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        from inference.embedder import Embedder
        import modal
        try:
            return Embedder().encode.remote([text])[0]
        except Exception:
            return modal.Cls.from_name("dev_fleet", "Embedder")().encode.remote([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        from inference.embedder import Embedder
        import modal
        try:
            return Embedder().encode.remote(texts)
        except Exception:
            return modal.Cls.from_name("dev_fleet", "Embedder")().encode.remote(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        from inference.embedder import Embedder
        import modal
        try:
            result = await Embedder().encode.remote.aio([query])
        except Exception:
            result = await modal.Cls.from_name("dev_fleet", "Embedder")().encode.remote.aio([query])
        return result[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        from inference.embedder import Embedder
        import modal
        try:
            result = await Embedder().encode.remote.aio([text])
        except Exception:
            result = await modal.Cls.from_name("dev_fleet", "Embedder")().encode.remote.aio([text])
        return result[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        from inference.embedder import Embedder
        import modal
        try:
            return await Embedder().encode.remote.aio(texts)
        except Exception:
            return await modal.Cls.from_name("dev_fleet", "Embedder")().encode.remote.aio(texts)

Settings.embed_model = ModalEmbeddings()

# Configure custom LlamaIndex LLM to use Modal vLLM
from orchestrator.llm_client import ModalVLLM
Settings.llm = ModalVLLM()


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
    """Container for three distinct knowledge graphs and a LlamaIndex PropertyGraph."""

    semantic: nx.DiGraph = field(default_factory=nx.DiGraph)
    procedural: nx.DiGraph = field(default_factory=nx.DiGraph)
    episodic: nx.DiGraph = field(default_factory=nx.DiGraph)

    # Exclude property_graph from normal dataclass initialization
    property_graph: Any = field(default=None, repr=False, init=False)

    def __post_init__(self):
        # Initialize an empty Property Graph using the global HF embeddings and Modal vLLM.
        self.property_graph = PropertyGraphIndex.from_documents([])

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

        # After loading networkx, rebuild PropertyGraph from Semantic/Procedural
        mem._rebuild_property_graph()
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
        # Also ingest into property graph
        self._ingest_node_to_pg(node_id, attrs, "semantic")

    # -- Procedural helpers --------------------------------------------------

    def add_procedural_node(
        self, node_id: str, attrs: dict[str, Any]
    ) -> None:
        self.procedural.add_node(node_id, **attrs)
        # Also ingest into property graph
        self._ingest_node_to_pg(node_id, attrs, "procedural")


    # -- Property Graph ingestion --------------------------------------------

    def _ingest_node_to_pg(self, node_id: str, attrs: dict[str, Any], graph_type: str):
        """Ingest explicitly constructed EntityNodes to bypass LLM extraction."""
        from llama_index.core.graph_stores.types import KG_NODES_KEY
        from llama_index.core.schema import TextNode

        if self.property_graph is None:
            self.property_graph = PropertyGraphIndex.from_existing(
                property_graph_store=SimplePropertyGraphStore(),
                kg_extractors=[],
            )

        text_content = f"[{graph_type.capitalize()}] {node_id}: {json.dumps(attrs, default=str)}"
        llama_node = TextNode(text=text_content, metadata={"node_id": node_id, "graph_type": graph_type})

        # Pass the extracted entity directly into LlamaIndex bypassing the extractors
        entity_node = EntityNode(
            name=node_id,
            label=graph_type,
            properties={"node_id": node_id, "graph_type": graph_type, **attrs}
        )
        llama_node.metadata[KG_NODES_KEY] = [entity_node]
        self.property_graph.insert_nodes([llama_node])

    def _rebuild_property_graph(self):
        from llama_index.core.graph_stores.types import KG_NODES_KEY
        from llama_index.core.schema import TextNode

        self.property_graph = PropertyGraphIndex.from_existing(
            property_graph_store=SimplePropertyGraphStore(),
            kg_extractors=[],
        )

        nodes_to_insert = []

        for node, data in self.semantic.nodes(data=True):
            text_content = f"[Semantic] {node}: {json.dumps(data, default=str)}"
            llama_node = TextNode(text=text_content, metadata={"node_id": node, "graph_type": "semantic"})
            entity_node = EntityNode(
                name=node,
                label="semantic",
                properties={"node_id": node, "graph_type": "semantic", **data}
            )
            llama_node.metadata[KG_NODES_KEY] = [entity_node]
            nodes_to_insert.append(llama_node)

        for node, data in self.procedural.nodes(data=True):
            text_content = f"[Procedural] {node}: {json.dumps(data, default=str)}"
            llama_node = TextNode(text=text_content, metadata={"node_id": node, "graph_type": "procedural"})
            entity_node = EntityNode(
                name=node,
                label="procedural",
                properties={"node_id": node, "graph_type": "procedural", **data}
            )
            llama_node.metadata[KG_NODES_KEY] = [entity_node]
            nodes_to_insert.append(llama_node)

        if nodes_to_insert:
            self.property_graph.insert_nodes(nodes_to_insert)

    # -- GraphRAG context builder --------------------------------------------

    def build_context(self, episodic_node_id: str) -> str:
        """Build a localized text context for *episodic_node_id*.

        Configures a LlamaIndex retriever that queries the PropertyGraphIndex.
        Uses the existing Qwen3-Reranker cross-encoder as a Node Postprocessor.
        """
        parts: list[str] = []

        # Episodic node itself
        if episodic_node_id in self.episodic:
            attrs = dict(self.episodic.nodes[episodic_node_id])
            parts.append(f"[Episodic] {episodic_node_id}: {json.dumps(attrs, default=str)}")

            # Use LlamaIndex property graph retriever to fetch related semantic/procedural
            if self.property_graph:
                query = str(attrs.get("description", ""))

                retriever = self.property_graph.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(query)

                # Filter using Qwen3-Reranker cross-encoder as a Node Postprocessor logic
                from orchestrator.rerank_engine import rerank_candidates
                candidates = []
                for node in retrieved_nodes:
                    node_id = node.metadata.get("node_id", "unknown")
                    candidates.append({"id": node_id, "description": node.text})

                edges = rerank_candidates(episodic_node_id, query, candidates)
                valid_ids = {e.candidate_id for e in edges}

                for node in retrieved_nodes:
                    node_id = node.metadata.get("node_id", "unknown")
                    graph_type = node.metadata.get("graph_type", "unknown")
                    if node_id in valid_ids:
                        parts.append(f"[{graph_type.capitalize()}] {node_id}: {node.text}")

            # Fallback for old manual networkx edges in case of test fixtures or incomplete retrieval
            for _, target, edge_data in self.episodic.out_edges(episodic_node_id, data=True):
                target_graph = edge_data.get("graph")

                if target_graph == "semantic" and target in self.semantic:
                    target_attrs = dict(self.semantic.nodes[target])
                    parts.append(f"[Semantic] {target}: {json.dumps(target_attrs, default=str)}")
                elif target_graph == "procedural" and target in self.procedural:
                    target_attrs = dict(self.procedural.nodes[target])
                    parts.append(f"[Procedural] {target}: {json.dumps(target_attrs, default=str)}")

        return "\n".join(parts) if parts else "(no context)"

    def to_dict(self) -> dict[str, Any]:
        """Export all three graphs as a JSON-serializable dictionary."""
        return {
            "semantic": nx.node_link_data(self.semantic),
            "procedural": nx.node_link_data(self.procedural),
            "episodic": nx.node_link_data(self.episodic),
        }

def generate_interactive_graph_html(memory: TriGraphMemory) -> str:
    """Merge TriGraph memory graphs and render PyVis network HTML."""
    from pyvis.network import Network
    import tempfile

    net = Network(height="600px", width="100%", directed=True, notebook=False)

    # Merge Semantic (Blue)
    for node, data in memory.semantic.nodes(data=True):
        label = data.get("name") or data.get("description") or str(node)
        # truncation
        if len(label) > 20:
            label = label[:17] + "..."
        net.add_node(str(node), label=label, color="#3498db", title=json.dumps(data, indent=2))

    for u, v, data in memory.semantic.edges(data=True):
        net.add_edge(str(u), str(v), title=json.dumps(data))

    # Merge Procedural (Green)
    for node, data in memory.procedural.nodes(data=True):
        label = data.get("rule") or data.get("description") or str(node)
        if len(label) > 20:
            label = label[:17] + "..."
        net.add_node(str(node), label=label, color="#2ecc71", title=json.dumps(data, indent=2))

    for u, v, data in memory.procedural.edges(data=True):
        net.add_edge(str(u), str(v), title=json.dumps(data))

    # Merge Episodic (Red)
    for node, data in memory.episodic.nodes(data=True):
        label = data.get("description") or str(node)
        if len(label) > 20:
            label = label[:17] + "..."
        net.add_node(str(node), label=label, color="#e74c3c", title=json.dumps(data, indent=2))

    for u, v, data in memory.episodic.edges(data=True):
        net.add_edge(str(u), str(v), title=json.dumps(data))

    net.toggle_physics(True)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        net.write_html(f.name)
        with open(f.name, "r") as html_f:
            html = html_f.read()
    os.remove(f.name)
    return html
