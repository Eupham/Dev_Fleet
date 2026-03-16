"""Tri-Graph Memory — Semantic, Procedural, and Episodic graphs.

Each graph is a ``networkx.DiGraph``.  Utility helpers merge the current
Episodic node with relevant Semantic/Procedural neighbours into a
localized text representation (GraphRAG context) that is sent to the 7B
model for generation.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core import Document, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.property_graph import VectorContextRetriever


# ---------------------------------------------------------------------------
# Zero-Trust Custom Embeddings using Modal RPC
# ---------------------------------------------------------------------------

class ModalEmbeddings(BaseEmbedding):
    """Routes embedding requests to the isolated Modal Embedder service."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


# ---------------------------------------------------------------------------
# Persistence paths (inside the mounted Modal Volume at /state)
# ---------------------------------------------------------------------------

STATE_DIR = Path(os.getenv("DEVFLEET_STATE_DIR", "/state"))

SEMANTIC_PATH = STATE_DIR / "semantic_graph.json"
PROCEDURAL_PATH = STATE_DIR / "procedural_graph.json"
EPISODIC_PATH = STATE_DIR / "episodic_graph.json"
PROP_GRAPH_DIR = STATE_DIR / "property_graph"

_GRAPH_PATHS = (SEMANTIC_PATH, PROCEDURAL_PATH, EPISODIC_PATH)


# ---------------------------------------------------------------------------
# In-memory singleton cache with mtime-based dirty flag
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cached_instance: Optional["TriGraphMemory"] = None
# Maps each path to its mtime at the time the cache was last populated.
_cached_mtimes: Dict[Path, float] = {}


def _current_mtimes() -> Dict[Path, float]:
    """Return the current modification timestamps for the three graph files."""
    return {p: p.stat().st_mtime if p.exists() else 0.0 for p in _GRAPH_PATHS}


def _cache_is_valid() -> bool:
    """Return True when every graph file mtime matches what is stored in the cache."""
    if _cached_instance is None:
        return False
    current = _current_mtimes()
    return all(current[p] == _cached_mtimes.get(p, -1.0) for p in _GRAPH_PATHS)


def _invalidate_cache() -> None:
    """Evict the singleton cache (called after save())."""
    global _cached_instance, _cached_mtimes
    with _cache_lock:
        _cached_instance = None
        _cached_mtimes = {}


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
        # property_graph is lazily initialized on first access via _ensure_property_graph().
        # LlamaIndex global settings must be configured separately via configure() before
        # any LlamaIndex call — see load() which calls configure() before cls().
        self.property_graph = None

    # -- Serialization -------------------------------------------------------

    def save(self) -> None:
        """Persist all three graphs as JSON node-link data and evict the singleton cache."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        for graph, path in [
            (self.semantic, SEMANTIC_PATH),
            (self.procedural, PROCEDURAL_PATH),
            (self.episodic, EPISODIC_PATH),
        ]:
            data = nx.node_link_data(graph, edges="edges")
            path.write_text(json.dumps(data, default=str))
        # Persist PropertyGraph (embeddings included) to avoid recomputation on load.
        if self.property_graph is not None:
            try:
                PROP_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
                self.property_graph.storage_context.persist(persist_dir=str(PROP_GRAPH_DIR))
            except Exception as _pg_exc:
                import logging
                logging.getLogger("dev_fleet.graph_memory").warning(
                    "PropertyGraph persist failed (%s) — embeddings will be recomputed on next load.", _pg_exc
                )
        # Evict the cache so the next load() reads the freshly written files.
        _invalidate_cache()

    def _ensure_property_graph(self) -> Any:
        """Lazy-initialize the PropertyGraph on first access.

        Must only be called after configure() has been invoked so that
        LlamaIndex Settings are pre-configured with our Modal models.
        """
        if self.property_graph is None:
            self.property_graph = PropertyGraphIndex.from_existing(
                property_graph_store=SimplePropertyGraphStore(),
                kg_extractors=[],
            )
        return self.property_graph

    @classmethod
    def configure(cls) -> None:
        """Pre-configure LlamaIndex global settings before instantiation.

        Must be called before :meth:`load` in contexts where LlamaIndex has not
        yet been initialised, to avoid the lazy-getter bootstrap that tries to
        reach OpenAI when no ``OPENAI_API_KEY`` is present. The only safe
        operation is a blind write.
        """
        from llama_index.core import Settings as LISettings
        from orchestrator.llm_client import ModalVLLM
        LISettings.embed_model = ModalEmbeddings()
        LISettings.llm = ModalVLLM()
        # Force batched processing to drastically reduce Modal RPC network latency
        LISettings.embed_batch_size = 128

    @classmethod
    def load(cls) -> "TriGraphMemory":
        """Restore graphs from disk, or return the in-memory singleton when the
        underlying files have not changed since the last load.

        This prevents the O(|V|+|E|) _rebuild_property_graph() from firing on
        every LangGraph node invocation when state is unchanged.
        """
        global _cached_instance, _cached_mtimes

        with _cache_lock:
            if _cache_is_valid():
                return _cached_instance  # type: ignore[return-value]

            # Configure LlamaIndex global settings before instantiation so that
            # __post_init__ (which no longer sets globals) is safe to call without
            # an OPENAI_API_KEY present.
            cls.configure()
            mem = cls()
            for attr, path in [
                ("semantic", SEMANTIC_PATH),
                ("procedural", PROCEDURAL_PATH),
                ("episodic", EPISODIC_PATH),
            ]:
                if path.exists():
                    data = json.loads(path.read_text())
                    # Normalise persisted data: files written before NX 3.4 used "links", not "edges"
                    if "links" in data and "edges" not in data:
                        data = {**data, "edges": data.pop("links")}
                    setattr(mem, attr, nx.node_link_graph(data, edges="edges"))

            # Restore PropertyGraph from disk if available; otherwise rebuild from NX graphs.
            # Loading from disk avoids O(N) remote embedding calls on every cold load.
            try:
                if PROP_GRAPH_DIR.exists():
                    from llama_index.core import StorageContext, load_index_from_storage
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(PROP_GRAPH_DIR)
                    )
                    mem.property_graph = load_index_from_storage(storage_context)
                else:
                    mem._rebuild_property_graph()
            except Exception as _load_exc:
                import logging
                logging.getLogger("dev_fleet.graph_memory").warning(
                    "PropertyGraph restore failed (%s) — rebuilding from NX graphs.", _load_exc
                )
                mem._rebuild_property_graph()

            # Populate the singleton cache.
            _cached_instance = mem
            _cached_mtimes = _current_mtimes()
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

    def add_semantic_node(self, node_id: str, attrs: dict[str, Any]) -> None:
        """Add a typed semantic node. Validates against SemanticNode schemas.
        Falls back to untyped insertion on ValidationError so the indexer
        does not break on nodes that predate the typed schema.
        """
        try:
            from orchestrator.node_schemas import SemanticNode
            from pydantic import TypeAdapter
            adapter = TypeAdapter(SemanticNode)
            validated = adapter.validate_python({**attrs, "node_id": node_id})
            self.semantic.add_node(node_id, **validated.model_dump())
        except Exception:
            self.semantic.add_node(node_id, **attrs)
        self._ingest_node_to_pg(
            node_id, dict(self.semantic.nodes[node_id]), "semantic"
        )

    def add_semantic_edge(
        self, src: str, dst: str, attrs: dict[str, Any] | None = None
    ) -> None:
        """Add a directed edge in the semantic graph and mirror it in the PropertyGraph."""
        self.semantic.add_edge(src, dst, **(attrs or {}))
        self._ingest_edge_to_pg(src, dst, (attrs or {}).get("relation", "related_to"))

    # -- Procedural helpers --------------------------------------------------

    def add_procedural_node(self, node_id: str, attrs: dict[str, Any]) -> None:
        """Add a typed procedural node. Falls back to untyped on ValidationError."""
        try:
            from orchestrator.node_schemas import ProceduralNode
            from pydantic import TypeAdapter
            adapter = TypeAdapter(ProceduralNode)
            validated = adapter.validate_python({**attrs, "node_id": node_id})
            self.procedural.add_node(node_id, **validated.model_dump())
        except Exception:
            self.procedural.add_node(node_id, **attrs)
        self._ingest_node_to_pg(
            node_id, dict(self.procedural.nodes[node_id]), "procedural"
        )

    def add_procedural_edge(
        self, src: str, dst: str, attrs: dict[str, Any] | None = None
    ) -> None:
        """Add a directed edge in the procedural graph and mirror it in the PropertyGraph."""
        self.procedural.add_edge(src, dst, **(attrs or {}))
        self._ingest_edge_to_pg(src, dst, (attrs or {}).get("relation", "leads_to"))


    # -- Property Graph ingestion --------------------------------------------

    def _ingest_node_to_pg(self, node_id: str, attrs: dict[str, Any], graph_type: str):
        """Ingest explicitly constructed EntityNodes to bypass LLM extraction.

        Calls the NodeLabeler to assign a typed label + content summary to each
        node.  The label/content are written back onto the NetworkX graph so
        they survive save/load cycles without a second LLM call.
        """
        from llama_index.core.graph_stores.types import KG_NODES_KEY
        from llama_index.core.schema import TextNode
        from orchestrator.labeler import label_node

        self._ensure_property_graph()

        # Assign label + content via the labeler (noop if already set).
        classification = label_node(node_id, attrs, graph_type)
        label = classification["label"]
        content = classification["content"]

        # Persist label/content back onto the NX node so _rebuild_property_graph
        # can reuse them without re-calling the LLM on every load.
        nx_graph = getattr(self, graph_type, None)
        if nx_graph is not None and node_id in nx_graph:
            nx_graph.nodes[node_id]["label"] = label
            nx_graph.nodes[node_id]["content"] = content

        enriched_attrs = {**attrs, "label": label, "content": content}
        text_content = f"[{label}] {node_id}: {content}"
        llama_node = TextNode(
            text=text_content,
            metadata={"node_id": node_id, "graph_type": graph_type, "label": label},
        )

        entity_node = EntityNode(
            name=node_id,
            label=label,
            properties={"node_id": node_id, "graph_type": graph_type, **enriched_attrs},
        )
        llama_node.metadata[KG_NODES_KEY] = [entity_node]
        self.property_graph.insert_nodes([llama_node])

    def _ingest_edge_to_pg(self, src: str, dst: str, label: str) -> None:
        """Upsert a single Relation edge into the LlamaIndex PropertyGraph store."""
        pg = self._ensure_property_graph()
        relation = Relation(source_id=src, target_id=dst, label=label)
        pg.property_graph_store.upsert_relation(relation)

    def _rebuild_property_graph(self):
        """Rebuild PropertyGraph from persisted NX graphs.

        Uses stored ``label`` / ``content`` attributes (written by _ingest_node_to_pg)
        so no LLM calls are needed here — labels survive across save/load cycles.
        """
        from llama_index.core.graph_stores.types import KG_NODES_KEY
        from llama_index.core.schema import TextNode

        self.property_graph = PropertyGraphIndex.from_existing(
            property_graph_store=SimplePropertyGraphStore(),
            kg_extractors=[],
        )

        nodes_to_insert = []

        for graph_type, nx_graph in [("semantic", self.semantic), ("procedural", self.procedural)]:
            for node, data in nx_graph.nodes(data=True):
                # Use stored label/content if present; fall back to graph_type name.
                label = data.get("label", graph_type.capitalize())
                content = data.get("content") or json.dumps(data, default=str)
                text_content = f"[{label}] {node}: {content}"
                llama_node = TextNode(
                    text=text_content,
                    metadata={"node_id": node, "graph_type": graph_type, "label": label},
                )
                entity_node = EntityNode(
                    name=node,
                    label=label,
                    properties={"node_id": node, "graph_type": graph_type, **data},
                )
                llama_node.metadata[KG_NODES_KEY] = [entity_node]
                nodes_to_insert.append(llama_node)

        if nodes_to_insert:
            self.property_graph.insert_nodes(nodes_to_insert)

        # Sync NetworkX structural edges as LlamaIndex Relation objects so the
        # PropertyGraph retriever can traverse them (true multi-hop GraphRAG).
        relations_to_insert = []

        for u, v, data in self.semantic.edges(data=True):
            relations_to_insert.append(
                Relation(source_id=u, target_id=v, label=data.get("relation", "related_to"))
            )

        for u, v, data in self.procedural.edges(data=True):
            relations_to_insert.append(
                Relation(source_id=u, target_id=v, label=data.get("relation", "leads_to"))
            )

        for rel in relations_to_insert:
            self.property_graph.property_graph_store.upsert_relation(rel)

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
            if (
                self.semantic.number_of_nodes() > 0 or self.procedural.number_of_nodes() > 0
            ):
                query = str(attrs.get("description", ""))
                try:
                    retriever = self.as_vector_retriever(similarity_top_k=5)
                    retrieved_nodes = retriever.retrieve(query)
                except Exception as exc:
                    import logging
                    logging.getLogger("dev_fleet.graph_memory").warning(
                        "PropertyGraph retrieval failed for %s (%s) — skipping.", episodic_node_id, exc
                    )
                    retrieved_nodes = []

                if retrieved_nodes:
                    from orchestrator.rerank_engine import rerank_candidates
                    candidates = []
                    for node in retrieved_nodes:
                        node_id = node.metadata.get("node_id", "unknown")
                        candidates.append({"id": node_id, "description": node.text})

                    try:
                        edges = rerank_candidates(episodic_node_id, query, candidates)
                        valid_ids = {e.candidate_id for e in edges}
                    except Exception:
                        valid_ids = {c["id"] for c in candidates}

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
                    frame = target_attrs.get("frame_name", "Semantic")
                    concept_type = target_attrs.get("concept_type", "")
                    content = target_attrs.get(
                        "content", target_attrs.get("description", str(target))
                    )
                    frame_label = f"{frame}({concept_type})" if concept_type else frame
                    parts.append(f"[{frame_label}] {target}: {content}")
                elif target_graph == "procedural" and target in self.procedural:
                    target_attrs = dict(self.procedural.nodes[target])
                    frame = target_attrs.get("frame_name", "Procedural")
                    actor = target_attrs.get("actor_capability", "")
                    depth = target_attrs.get("implementation_depth", "")
                    cost = target_attrs.get("execution_cost", "")
                    content = target_attrs.get(
                        "content", target_attrs.get("description", str(target))
                    )
                    detail = " / ".join(x for x in [actor, depth, cost] if x)
                    parts.append(
                        f"[{frame}({detail})] {target}: {content}" if detail
                        else f"[{frame}] {target}: {content}"
                    )

        return "\n".join(parts) if parts else "(no context)"

    def as_vector_retriever(self, similarity_top_k: int = 10):
        """Return an explicit VectorContextRetriever. Never falls back to LLM synonym retrieval."""
        pg = self._ensure_property_graph()
        return VectorContextRetriever(
            pg.property_graph_store,
            embed_model=Settings.embed_model,
            similarity_top_k=similarity_top_k,
            path_depth=1,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export all three graphs as a JSON-serializable dictionary."""
        return {
            "semantic": nx.node_link_data(self.semantic, edges="edges"),
            "procedural": nx.node_link_data(self.procedural, edges="edges"),
            "episodic": nx.node_link_data(self.episodic, edges="edges"),
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
