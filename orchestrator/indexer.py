# orchestrator/indexer.py
"""Codebase indexer — builds knowledge graphs from the local workspace.

Routes all file extraction through orchestrator/extractor.py so that
the indexer and onboard_domain share a single pipeline. Output is typed
node dicts validated against node_schemas.py before insertion.

Respects .gitignore via pathspec.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

import pathspec
from pydantic import TypeAdapter, ValidationError

from orchestrator.extractor import extract_from_artifact
from orchestrator.graph_memory import TriGraphMemory
from orchestrator.node_schemas import SemanticNode, ProceduralNode

logger = logging.getLogger("dev_fleet.indexer")

_SEM_ADAPTER = TypeAdapter(SemanticNode)
_PROC_ADAPTER = TypeAdapter(ProceduralNode)

# File extensions routed to the extractor
_CODE_SUFFIXES = frozenset({".py", ".js", ".ts", ".jsx", ".tsx"})
_PROSE_SUFFIXES = frozenset({".md", ".rst", ".txt", ".sh"})
_SKIP_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".so", ".egg", ".whl",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".zip", ".tar", ".gz", ".bz2",
    ".lock", ".sum",
})


def build_knowledge_graphs(workspace_dir: str) -> dict:
    """Index workspace_dir and insert typed nodes into the knowledge graphs.

    Returns:
        dict: semantic_nodes_added, procedural_nodes_added, files_processed
    """
    memory = TriGraphMemory.load()
    sem_count = proc_count = files = 0

    # Load .gitignore if present
    gitignore_path = os.path.join(workspace_dir, ".gitignore")
    ignore_spec = None
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            ignore_spec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, f
            )

    for root, dirs, filenames in os.walk(workspace_dir):
        # Prune ignored directories in-place
        if ignore_spec:
            rel_root = os.path.relpath(root, workspace_dir)
            dirs[:] = [
                d for d in dirs
                if not ignore_spec.match_file(
                    os.path.join(rel_root, d) + "/"
                )
            ]

        for filename in filenames:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, workspace_dir)
            suffix = Path(filename).suffix.lower()

            if suffix in _SKIP_SUFFIXES:
                continue
            if ignore_spec and ignore_spec.match_file(rel_path):
                continue
            if suffix not in _CODE_SUFFIXES and suffix not in _PROSE_SUFFIXES:
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as exc:
                logger.debug("Cannot read %s: %s", rel_path, exc)
                continue

            node_dicts = extract_from_artifact(source, rel_path)
            for node_dict in node_dicts:
                graph_type = node_dict.get("graph_type", "")
                node_id = node_dict.get("node_id", "")
                if not node_id:
                    continue
                if graph_type == "semantic":
                    try:
                        validated = _SEM_ADAPTER.validate_python(node_dict)
                        if validated.node_id not in memory.semantic:
                            memory.add_semantic_node(
                                validated.node_id, validated.model_dump()
                            )
                            sem_count += 1
                    except (ValidationError, Exception):
                        pass
                elif graph_type == "procedural":
                    try:
                        validated = _PROC_ADAPTER.validate_python(node_dict)
                        if validated.node_id not in memory.procedural:
                            memory.add_procedural_node(
                                validated.node_id, validated.model_dump()
                            )
                            proc_count += 1
                    except (ValidationError, Exception):
                        pass

            files += 1

    memory.save()
    result = {
        "semantic_nodes_added": sem_count,
        "procedural_nodes_added": proc_count,
        "files_processed": files,
    }
    logger.info("Indexer: %s", result)
    return result
