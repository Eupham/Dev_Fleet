import ast
import os
import pathspec
from llama_index.core.node_parser import SentenceSplitter
from orchestrator.graph_memory import TriGraphMemory

def build_knowledge_graphs(workspace_dir: str):
    memory = TriGraphMemory.load()

    # Parse .gitignore
    gitignore_path = os.path.join(workspace_dir, ".gitignore")
    ignore_spec = None
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            ignore_spec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, f
            )

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)

    for root, dirs, files in os.walk(workspace_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, workspace_dir)

            if ignore_spec and ignore_spec.match_file(rel_path):
                continue

            # Check if directory should be ignored
            # Pathspec expects directories to have a trailing slash, or we check prefixes
            rel_dir = os.path.relpath(root, workspace_dir)
            if ignore_spec and rel_dir != "." and ignore_spec.match_file(rel_dir + "/"):
                # We should have pruned dirs, but os.walk continues unless we modify `dirs` in place.
                # So we can just skip file processing here.
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception:
                continue  # skip unreadable or binary files

            if file_path.endswith(".py"):
                # Semantic Graph
                try:
                    tree = ast.parse(file_content)
                except SyntaxError:
                    continue

                memory.add_semantic_node(rel_path, {"type": "File"})

                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                        docstring = ast.get_docstring(node) or ""
                        node_id = f"{rel_path}::{node.name}"

                        memory.add_semantic_node(node_id, {
                            "type": type(node).__name__,
                            "name": node.name,
                            "docstring": docstring,
                            "lineno": node.lineno
                        })
                        memory.add_semantic_edge(rel_path, node_id, {"relation": "defines"})

            elif file_path.endswith((".md", ".sh", ".txt")):
                # Procedural Graph
                chunks = splitter.split_text(file_content)
                for i, chunk in enumerate(chunks):
                    node_id = f"{rel_path}_chunk_{i}"
                    memory.add_procedural_node(node_id, {
                        "type": "Chunk",
                        "content": chunk
                    })

                    if i > 0:
                        prev_node_id = f"{rel_path}_chunk_{i-1}"
                        memory.add_procedural_edge(prev_node_id, node_id, {"relation": "leads_to"})

    memory.save()
