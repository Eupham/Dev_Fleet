# orchestrator/extractor.py
"""Extraction of typed knowledge graph nodes from code, HTML, PDFs, and prose.

Entry point: extract_from_artifact(text, filename, content_hint) -> list[dict]

Routes to:
  _extract_python()      for .py files        — stdlib ast, zero extra deps
  _extract_javascript()  for .js/.ts           — tree-sitter
  _extract_html()        for HTML content      — trafilatura XML, heading hierarchy
  _extract_pdf()         for .pdf files/bytes  — pymupdf TOC + font-size fallback
  _extract_markdown()    for .md/.rst and text — markdown-it-py
  _extract_prose()       fallback              — chunked LLM extraction

HTML extraction via trafilatura:
  bare_extraction(html, output_format="xml", include_formatting=True,
                  include_links=True, deduplicate=True)
  Returns an XML string. Heading elements are <head rend="h1">, <head rend="h2">,
  etc. Parsing this XML with lxml gives the heading hierarchy directly — no
  heuristics needed for clean HTML. Outbound links are preserved in <ref target="">
  elements and registered as graph edges.

PDF extraction via pymupdf:
  Strategy 1 (preferred): doc.get_toc() returns [[level, title, page], ...]
  when the PDF has an embedded table of contents. This is an exact hierarchy,
  no inference needed. Applies to most well-structured technical documents.

  Strategy 2 (fallback): page.get_text("dict") returns per-span font sizes.
  Lines whose max font size is significantly larger than the document's modal
  body font size are classified as headings. Level is inferred from font size
  rank. Text between headings is the section body.

Chunking:
  Code files: not chunked. Parsers handle full structure.
  HTML/PDF sections: individual sections passed to LLM when needed.
  Prose: _chunk_text() produces overlapping windows fed to LLM calls.

LLM calls are isolated to three narrow functions:
  _llm_summarize()              — one-sentence content when docstring absent
  _llm_classify_concept_type()  — concept_type when AST is ambiguous
  _llm_extract_constraints()    — ConstraintNode fields from a single prose chunk
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dev_fleet.extractor")

_CHUNK_SIZE = 1800
_CHUNK_OVERLAP = 200

# Heading threshold: a line is a heading if its font size exceeds body size by
# this fraction. Tunable — 0.15 means 15% larger than modal body font.
_PDF_HEADING_SIZE_THRESHOLD = 0.15

try:
    from tree_sitter import Language, Parser as TSParser
    import tree_sitter_javascript as tsjs
    _JS_LANGUAGE = Language(tsjs.language())
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

try:
    from markdown_it import MarkdownIt
    _MD_AVAILABLE = True
except ImportError:
    _MD_AVAILABLE = False

try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False

try:
    import pymupdf
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# Known pattern names for class name heuristic
_PATTERN_NAMES = frozenset({
    "singleton", "factory", "observer", "strategy", "command",
    "decorator", "adapter", "facade", "proxy", "builder",
    "iterator", "visitor", "mediator", "memento", "state",
    "template", "chain", "composite", "registry",
})

# Stdlib module names — used to determine implementation_depth
_STDLIB = frozenset({
    "os", "sys", "re", "json", "ast", "math", "collections", "itertools",
    "functools", "pathlib", "typing", "dataclasses", "abc", "io", "time",
    "datetime", "logging", "threading", "asyncio", "uuid", "hashlib",
    "base64", "copy", "weakref", "contextlib", "inspect", "types",
    "string", "struct", "array", "queue", "heapq", "bisect", "enum",
    "traceback", "warnings", "unittest", "tempfile", "shutil", "glob",
    "subprocess", "socket", "ssl", "http", "urllib", "email", "html",
    "xml", "csv", "configparser", "argparse", "pickle", "shelve",
    "sqlite3", "zlib", "gzip", "bz2", "lzma", "zipfile", "tarfile",
})

_SYSCALL_MODULES = frozenset({"ctypes", "cffi", "mmap", "fcntl", "termios"})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_from_artifact(
    text: str,
    filename: str = "",
    content_hint: str = "",
) -> list[dict]:
    """Extract typed knowledge graph node dicts from an artifact.

    Args:
        text:         Raw text content of the artifact.
        filename:     Original filename or URL. Used to select extractor.
                      For URLs, the path component is used for extension
                      detection. Pass empty string for unknown sources.
        content_hint: Optional MIME type hint, e.g. "text/html".
                      Takes precedence over filename extension when provided.

    Returns:
        List of dicts compatible with SemanticNode or ProceduralNode schemas.
        Empty list if text is empty or no nodes could be extracted.
    """
    if not text or not text.strip():
        return []

    # Determine effective filename for extension routing
    # For URLs, use only the path component (strip query strings and fragments)
    clean_name = filename.split("?")[0].split("#")[0] if filename else ""
    suffix = Path(clean_name).suffix.lower() if clean_name else ""

    # Content hint overrides extension
    if content_hint:
        hint = content_hint.lower()
        if "html" in hint:
            return _extract_html(text, filename)
        if "javascript" in hint or "typescript" in hint:
            return _extract_javascript(text, filename)
        if "python" in hint:
            return _extract_python(text, filename)

    # Extension-based routing for local files and URLs with clean paths
    if suffix == ".py":
        return _extract_python(text, filename)
    if suffix in (".js", ".ts", ".jsx", ".tsx"):
        return _extract_javascript(text, filename)
    if suffix in (".md", ".rst"):
        return _extract_markdown(text, filename)
    if suffix == ".pdf":
        return _extract_pdf(text, filename)

    # Content sniffing for extensionless or ambiguous sources
    # HTML detection: presence of opening tags is a reliable signal
    stripped_start = text.lstrip()
    if stripped_start.startswith("<") and (
        "<html" in stripped_start[:500].lower()
        or "<body" in stripped_start[:500].lower()
        or "<!doctype" in stripped_start[:100].lower()
    ):
        return _extract_html(text, filename)

    # .txt and unknown extensions: attempt markdown structure,
    # fall back to prose chunking
    return _extract_markdown(text, filename) if _MD_AVAILABLE else _extract_prose(text)


# ---------------------------------------------------------------------------
# HTML extraction — BeautifulSoup strip then markdown pipeline
# ---------------------------------------------------------------------------

def _extract_html(html: str, source_name: str = "") -> list[dict]:
    """Extract nodes from HTML using trafilatura's structured XML output.

    trafilatura.bare_extraction with output_format="xml" returns an XML
    string where headings are <head rend="h1">, <head rend="h2"> etc.
    Parsing this with lxml gives the heading hierarchy directly.
    Outbound links are in <ref target="..."> elements.

    Falls back to regex stripping if trafilatura is unavailable.
    """
    if not _TRAFILATURA_AVAILABLE:
        # Minimal regex fallback — convert heading tags to markdown markers
        text = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            lambda m: "#" * int(m.group(1)) + " " + re.sub(r"<[^>]+>", "", m.group(2)),
            html, flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text).strip()
        return _extract_markdown(text, source_name) if _MD_AVAILABLE else _extract_prose(text)

    xml_str = trafilatura.extract(
        html,
        output_format="xml",
        include_formatting=True,
        include_links=True,
        deduplicate=True,
        favor_recall=True,
    )
    if not xml_str:
        return []

    return _parse_trafilatura_xml(xml_str, source_name)


def _parse_trafilatura_xml(xml_str: str, source_name: str) -> list[dict]:
    """Parse trafilatura XML output into typed graph nodes.

    Heading structure:
      <head rend="h1">Title</head>  → level 1
      <head rend="h2">Sub</head>    → level 2
    Paragraphs between headings accumulate as section body.
    Outbound links in <ref target="..."> become graph edge hints.

    Hierarchy is built by tracking the current heading stack. A heading
    at level N pops all stack entries at level >= N before pushing.
    The parent node_id of each section is the top of the stack after
    the pop — this gives the parent edge without any heuristics.
    """
    try:
        from lxml import etree
    except ImportError:
        # lxml not available — fall back to text extraction
        text = re.sub(r"<[^>]+>", " ", xml_str)
        return _extract_markdown(text, source_name) if _MD_AVAILABLE else []

    source_id = re.sub(r"[^a-zA-Z0-9_]", "_", source_name or "web")
    nodes: list[dict] = []
    outbound_links: list[str] = []

    try:
        root = etree.fromstring(xml_str.encode("utf-8"))
    except etree.XMLSyntaxError:
        return []

    # Stack entries: (level: int, node_id: str)
    heading_stack: list[tuple[int, str]] = []
    current_body_parts: list[str] = []
    current_heading_text: str = ""
    current_heading_level: int = 0
    current_node_id: str = ""

    def _flush_current_section():
        """Emit the accumulated section as a ConceptNode."""
        if not current_node_id:
            return
        body = " ".join(current_body_parts).strip()
        content = body[:200] if body else current_heading_text[:200]
        if not content:
            return
        node: dict = {
            "frame_name": "Code_Concept",
            "node_id": current_node_id,
            "content": content,
            "label": "Concept",
            "graph_type": "semantic",
            "concept_type": "pattern",
            "source_file": source_name,
            "language": "prose",
        }
        # Parent edge hint — stored as a non-schema field;
        # graph_memory will create the edge if parent_node_id is present
        if heading_stack:
            node["parent_node_id"] = heading_stack[-1][1]
        nodes.append(node)
        # Check for constraint-pattern headings
        if current_heading_text.lower() in _CONSTRAINT_HEADINGS and body:
            constraints = _llm_extract_constraints_chunked(body, current_heading_text)
            for idx, c in enumerate(constraints):
                c.setdefault("node_id", f"{current_node_id}__c{idx}")
                c.setdefault("label", "Execution_Constraint")
                c.setdefault("graph_type", "procedural")
                c.setdefault("content", c.get("enforced_by", "")[:200])
                nodes.append(c)

    for elem in root.iter():
        tag = etree.QName(elem.tag).localname if "{" in str(elem.tag) else elem.tag

        if tag == "head":
            # Flush previous section before starting new heading
            _flush_current_section()
            current_body_parts = []

            rend = elem.get("rend", "h2")
            level = int(rend[-1]) if rend and rend[-1].isdigit() else 2
            heading_text = (elem.text or "").strip()
            if not heading_text:
                continue

            # Pop stack to maintain proper nesting
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            node_id = (
                f"{source_id}__"
                + re.sub(r"[^a-zA-Z0-9_]", "_", heading_text)[:60]
                + f"_L{level}"
            )
            current_heading_text = heading_text
            current_heading_level = level
            current_node_id = node_id
            heading_stack.append((level, node_id))

        elif tag in ("p", "list", "quote", "code"):
            text_parts = [elem.text or ""]
            for child in elem:
                text_parts.append(child.text or "")
                text_parts.append(child.tail or "")
            current_body_parts.append(" ".join(text_parts).strip())

        elif tag == "ref":
            href = elem.get("target", "")
            if href.startswith("http"):
                outbound_links.append(href)

    _flush_current_section()

    # Attach outbound links to the first node as metadata
    if nodes and outbound_links:
        nodes[0]["outbound_links"] = outbound_links[:50]

    return nodes


def _extract_pdf(source, filepath: str = "") -> list[dict]:
    """Extract nodes from a PDF using pymupdf.

    Strategy 1 — embedded TOC (preferred):
      doc.get_toc() returns [[level, title, page], ...].
      When non-empty, this is the document's own declared structure.
      Each entry becomes a ConceptNode. Parent edges follow from level.
      Section body text is extracted from the page range between
      consecutive TOC entries at the same or lower level.

    Strategy 2 — font-size heuristic (fallback):
      page.get_text("dict") returns per-span font sizes.
      The modal body font size is the most frequent size across the doc.
      Lines whose max span size exceeds body size by _PDF_HEADING_SIZE_THRESHOLD
      are classified as headings. Level is inferred from font-size rank.

    Args:
        source: Either a file path string, bytes object, or raw text string
                from a .pdf file. Bytes are passed directly to pymupdf.open.
    """
    if not _PYMUPDF_AVAILABLE:
        logger.warning("pymupdf not available — PDF extraction skipped for %s", filepath)
        return []

    source_id = re.sub(r"[^a-zA-Z0-9_]", "_", filepath or "pdf_doc")
    nodes: list[dict] = []

    try:
        if isinstance(source, (bytes, bytearray)):
            doc = pymupdf.open(stream=source, filetype="pdf")
        elif isinstance(source, str) and source.startswith("%PDF"):
            doc = pymupdf.open(stream=source.encode("latin-1"), filetype="pdf")
        else:
            doc = pymupdf.open(source)
    except Exception as exc:
        logger.warning("pymupdf.open failed for %s: %s", filepath, exc)
        return []

    with doc:
        toc = doc.get_toc()

        if toc:
            # Strategy 1: embedded TOC — exact hierarchy
            nodes = _pdf_nodes_from_toc(doc, toc, source_id, filepath)
        else:
            # Strategy 2: font-size heuristics
            nodes = _pdf_nodes_from_fonts(doc, source_id, filepath)

    return nodes


def _pdf_nodes_from_toc(doc, toc: list, source_id: str, filepath: str) -> list[dict]:
    """Build ConceptNodes from embedded PDF table of contents.

    toc format: [[level, title, page_number], ...]
    level is 1-indexed. Parent is the closest preceding entry at level-1.
    """
    nodes: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, node_id)

    for i, entry in enumerate(toc):
        level, title, page = entry[0], entry[1], entry[2]
        title = title.strip()
        if not title:
            continue

        # Extract body text from this page to the next entry's page
        next_page = toc[i + 1][2] if i + 1 < len(toc) else page + 3
        body_parts = []
        for pg_num in range(max(0, page - 1), min(len(doc), next_page)):
            page_text = doc[pg_num].get_text("text")
            # Remove the heading text itself from the body
            body_parts.append(page_text.replace(title, "", 1).strip())
        body = " ".join(body_parts)[:500]

        node_id = (
            f"{source_id}__"
            + re.sub(r"[^a-zA-Z0-9_]", "_", title)[:60]
            + f"_p{page}"
        )

        while stack and stack[-1][0] >= level:
            stack.pop()

        node: dict = {
            "frame_name": "Code_Concept",
            "node_id": node_id,
            "content": body[:200] if body.strip() else title,
            "label": "Concept",
            "graph_type": "semantic",
            "concept_type": "pattern",
            "source_file": filepath,
            "language": "prose",
        }
        if stack:
            node["parent_node_id"] = stack[-1][1]
        nodes.append(node)
        stack.append((level, node_id))

        # Constraint extraction for relevant headings
        if title.lower() in _CONSTRAINT_HEADINGS and body.strip():
            constraints = _llm_extract_constraints_chunked(body, title)
            for idx, c in enumerate(constraints):
                c.setdefault("node_id", f"{node_id}__c{idx}")
                c.setdefault("label", "Execution_Constraint")
                c.setdefault("graph_type", "procedural")
                c.setdefault("content", c.get("enforced_by", "")[:200])
                nodes.append(c)

    return nodes


def _pdf_nodes_from_fonts(doc, source_id: str, filepath: str) -> list[dict]:
    """Infer heading structure from font sizes when no embedded TOC exists.

    Computes the modal body font size across the document.
    Lines exceeding that size by _PDF_HEADING_SIZE_THRESHOLD are headings.
    Level is assigned by ranking distinct heading sizes (largest = level 1).
    """
    from collections import Counter

    # Collect all font sizes across the document
    size_counts: Counter = Counter()
    all_lines: list[dict] = []  # {"text", "size", "page"}

    for page in doc:
        blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                max_size = max(s["size"] for s in spans)
                size_counts[round(max_size, 1)] += 1
                all_lines.append({"text": text, "size": max_size, "page": page.number})

    if not size_counts:
        return []

    body_size = size_counts.most_common(1)[0][0]
    heading_threshold = body_size * (1 + _PDF_HEADING_SIZE_THRESHOLD)

    # Rank distinct heading sizes largest-first → level assignment
    heading_sizes = sorted(
        {round(s, 1) for s in size_counts if s >= heading_threshold},
        reverse=True,
    )
    size_to_level = {sz: i + 1 for i, sz in enumerate(heading_sizes)}

    # Build sections
    nodes: list[dict] = []
    stack: list[tuple[int, str]] = []
    current_node_id: str = ""
    current_heading: str = ""
    current_body: list[str] = []

    def _flush():
        if not current_node_id:
            return
        body = " ".join(current_body).strip()
        node: dict = {
            "frame_name": "Code_Concept",
            "node_id": current_node_id,
            "content": body[:200] if body else current_heading,
            "label": "Concept",
            "graph_type": "semantic",
            "concept_type": "pattern",
            "source_file": filepath,
            "language": "prose",
        }
        if stack:
            node["parent_node_id"] = stack[-1][1]
        nodes.append(node)
        if current_heading.lower() in _CONSTRAINT_HEADINGS and body:
            for idx, c in enumerate(_llm_extract_constraints_chunked(body, current_heading)):
                c.setdefault("node_id", f"{current_node_id}__c{idx}")
                c.setdefault("label", "Execution_Constraint")
                c.setdefault("graph_type", "procedural")
                c.setdefault("content", c.get("enforced_by", "")[:200])
                nodes.append(c)

    for line_info in all_lines:
        text = line_info["text"]
        size = round(line_info["size"], 1)

        if size in size_to_level:
            _flush()
            current_body = []
            level = size_to_level[size]
            while stack and stack[-1][0] >= level:
                stack.pop()
            current_node_id = (
                f"{source_id}__"
                + re.sub(r"[^a-zA-Z0-9_]", "_", text)[:60]
                + f"_p{line_info['page']}"
            )
            current_heading = text
            stack.append((level, current_node_id))
        else:
            current_body.append(text)

    _flush()
    return nodes


# ---------------------------------------------------------------------------
# Text chunking utility
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
                overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level windows.

    Tries to break at sentence boundaries ('. ', '? ', '! ', '\n')
    within a tolerance window of 20% of chunk_size so chunks do not
    split mid-sentence. Falls back to hard break if no boundary found.

    Args:
        text:       Input text.
        chunk_size: Target characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks. Single-element list if text fits in one chunk.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    tolerance = chunk_size // 5

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to find a sentence break within the tolerance window
            search_start = max(start, end - tolerance)
            best_break = -1
            for sep in (". ", "? ", "! ", "\n\n", "\n"):
                idx = text.rfind(sep, search_start, end)
                if idx > best_break:
                    best_break = idx + len(sep)
            if best_break > start:
                end = best_break

        chunks.append(text[start:end])
        start = end - overlap

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Python extraction — stdlib ast
# ---------------------------------------------------------------------------

def _extract_python(source: str, filepath: str) -> list[dict]:
    """Extract nodes from Python source using the stdlib ast module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("ast.parse failed for %s: %s", filepath, exc)
        return []

    nodes: list[dict] = []
    rel_path = filepath or "unknown.py"

    # Collect module-level imports for implementation_depth classification
    module_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_imports.add(node.module.split(".")[0])

    # ModuleNode — always fully mechanical
    module_node_id = re.sub(r"[^a-zA-Z0-9_]", "_", rel_path)
    module_docstring = ast.get_docstring(tree) or ""
    exports = _get_exports(tree)
    depends_on = sorted(module_imports)
    content = module_docstring.split("\n")[0][:200] if module_docstring else ""
    if not content:
        content = _llm_summarize(module_node_id, f"Python module: {rel_path}")
    nodes.append({
        "frame_name": "Code_Module",
        "node_id": module_node_id,
        "content": content,
        "label": "Code_Module",
        "graph_type": "semantic",
        "module_path": rel_path,
        "exports": exports,
        "depends_on": depends_on,
        "language": "python",
    })

    # Process top-level classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            nodes.extend(_extract_python_class(node, rel_path, module_imports))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exec_node = _extract_python_function(node, rel_path, module_imports)
            if exec_node:
                nodes.append(exec_node)

    return nodes


def _get_exports(tree: ast.Module) -> list[str]:
    """Return __all__ contents or all public top-level names."""
    for node in ast.iter_child_nodes(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, ast.List)
        ):
            return [
                elt.s for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.s, str)
            ]
    public = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    public.append(target.id)
    return public


def _extract_python_class(
    node: ast.ClassDef, filepath: str, module_imports: set
) -> list[dict]:
    """Extract a ConceptNode from a class definition."""
    node_id = f"{re.sub(r'[^a-zA-Z0-9_]', '_', filepath)}__{node.name}"
    docstring = ast.get_docstring(node) or ""
    content = docstring.split("\n")[0][:200] if docstring else ""

    concept_type = _classify_python_concept_type(node)
    if concept_type is None:
        class_info = {
            "name": node.name,
            "bases": [
                b.id if isinstance(b, ast.Name) else
                b.attr if isinstance(b, ast.Attribute) else ""
                for b in node.bases
            ],
            "methods": [
                n.name for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ],
            "docstring": docstring[:300],
        }
        concept_type = _llm_classify_concept_type(node_id, class_info)

    if not content:
        content = _llm_summarize(
            node_id, f"Python class {node.name} in {filepath}: {docstring[:200]}"
        )

    return [{
        "frame_name": "Code_Concept",
        "node_id": node_id,
        "content": content,
        "label": concept_type,
        "graph_type": "semantic",
        "concept_type": concept_type,
        "source_file": filepath,
        "language": "python",
    }]


def _classify_python_concept_type(node: ast.ClassDef) -> Optional[str]:
    """Determine concept_type from AST structure. Returns None if ambiguous."""
    base_names = {
        b.id if isinstance(b, ast.Name) else
        b.attr if isinstance(b, ast.Attribute) else ""
        for b in node.bases
    }
    decorator_names = set()
    for d in node.decorator_list:
        if isinstance(d, ast.Name):
            decorator_names.add(d.id)
        elif isinstance(d, ast.Attribute):
            decorator_names.add(d.attr)
        elif isinstance(d, ast.Call):
            if isinstance(d.func, ast.Name):
                decorator_names.add(d.func.id)

    if "Protocol" in base_names:
        return "protocol"
    if base_names & {"ABC", "ABCMeta"}:
        return "interface"

    for item in ast.iter_child_nodes(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in item.decorator_list:
                dec_name = (
                    dec.id if isinstance(dec, ast.Name) else
                    dec.attr if isinstance(dec, ast.Attribute) else ""
                )
                if dec_name == "abstractmethod":
                    return "interface"

    if "dataclass" in decorator_names:
        return "data_structure"

    method_names = {
        item.name for item in ast.iter_child_nodes(node)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if {"__getitem__", "__len__"}.issubset(method_names):
        return "data_structure"
    if "__post_init__" in method_names:
        return "invariant"
    if any(p in node.name.lower() for p in _PATTERN_NAMES):
        return "pattern"

    return None


def _extract_python_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    filepath: str,
    module_imports: set,
) -> Optional[dict]:
    """Extract an ExecutionNode from a top-level function."""
    node_id = f"{re.sub(r'[^a-zA-Z0-9_]', '_', filepath)}__{node.name}"
    docstring = ast.get_docstring(node) or ""
    content = docstring.split("\n")[0][:200] if docstring else ""

    actor = _classify_actor_capability(node)
    depth = _classify_implementation_depth(node, module_imports)

    if not content:
        content = _llm_summarize(
            node_id, f"Python function {node.name} in {filepath}: {docstring[:200]}"
        )

    cost = "trivial"
    if depth == "algorithm":
        cost = "moderate"
    if depth == "syscall":
        cost = "expensive"

    return {
        "frame_name": "Task_Execution",
        "node_id": node_id,
        "content": content,
        "label": "Task_Execution",
        "graph_type": "procedural",
        "actor_capability": actor,
        "implementation_depth": depth,
        "execution_cost": cost,
    }


def _classify_actor_capability(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    body = node.body
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, (ast.Raise, ast.Pass)):
            return "llm_only"
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return "llm_only"

    _SHELL_FUNC_NAMES = frozenset({
        "system", "Popen", "run", "check_output", "check_call",
        "call", "getoutput", "getstatusoutput",
    })
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = (
                func.id if isinstance(func, ast.Name) else
                func.attr if isinstance(func, ast.Attribute) else ""
            )
            if name in _SHELL_FUNC_NAMES:
                return "bash"

    return "python"


def _classify_implementation_depth(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_imports: set,
) -> str:
    if module_imports & _SYSCALL_MODULES:
        return "syscall"

    non_stdlib = module_imports - _STDLIB
    has_loop = any(
        isinstance(child, (ast.For, ast.While))
        for child in ast.walk(node)
    )
    has_recursion = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == node.name
        for child in ast.walk(node)
    )

    if not non_stdlib and (has_loop or has_recursion):
        return "algorithm"
    return "library"


# ---------------------------------------------------------------------------
# JavaScript / TypeScript extraction — tree-sitter
# ---------------------------------------------------------------------------

def _extract_javascript(source: str, filepath: str) -> list[dict]:
    """Extract nodes from JS/TS source using tree-sitter."""
    if not _TS_AVAILABLE:
        logger.warning(
            "tree-sitter not available — falling back to prose for %s", filepath
        )
        return _extract_prose(source)

    parser = TSParser(_JS_LANGUAGE)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    nodes: list[dict] = []
    rel_path = filepath or "unknown.js"
    module_node_id = re.sub(r"[^a-zA-Z0-9_]", "_", rel_path)

    content = _llm_summarize(module_node_id, f"JS/TS module: {rel_path}")
    nodes.append({
        "frame_name": "Code_Module",
        "node_id": module_node_id,
        "content": content,
        "label": "Code_Module",
        "graph_type": "semantic",
        "module_path": rel_path,
        "language": "javascript",
    })

    def _walk(node):
        yield node
        for child in node.children:
            yield from _walk(child)

    for ts_node in _walk(tree.root_node):
        if ts_node.type == "class_declaration":
            name_node = next(
                (c for c in ts_node.children if c.type == "identifier"), None
            )
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode()
                node_id = f"{module_node_id}__{name}"
                class_src = source_bytes[ts_node.start_byte:ts_node.end_byte].decode()
                concept_type = _llm_classify_concept_type(
                    node_id, {"name": name, "source": class_src[:400]}
                )
                content = _llm_summarize(node_id, f"JS class {name} in {rel_path}")
                nodes.append({
                    "frame_name": "Code_Concept",
                    "node_id": node_id,
                    "content": content,
                    "label": concept_type,
                    "graph_type": "semantic",
                    "concept_type": concept_type,
                    "source_file": rel_path,
                    "language": "javascript",
                })

        elif ts_node.type in ("function_declaration", "arrow_function"):
            name_node = next(
                (c for c in ts_node.children if c.type == "identifier"), None
            )
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode()
                node_id = f"{module_node_id}__{name}"
                content = _llm_summarize(node_id, f"JS function {name} in {rel_path}")
                nodes.append({
                    "frame_name": "Task_Execution",
                    "node_id": node_id,
                    "content": content,
                    "label": "Task_Execution",
                    "graph_type": "procedural",
                    "actor_capability": "python",
                    "implementation_depth": "library",
                })

    return nodes


# ---------------------------------------------------------------------------
# Markdown / prose extraction — markdown-it-py
# ---------------------------------------------------------------------------

_CONSTRAINT_HEADINGS = frozenset({
    "requirements", "constraints", "limits", "restrictions",
    "rules", "security", "invariants", "preconditions",
    "postconditions", "guarantees", "assumptions",
})


def _extract_markdown(text: str, source_name: str = "") -> list[dict]:
    """Extract nodes from markdown or plain text using markdown-it-py.

    Headings become ConceptNode candidates.
    Sections under constraint-related headings are chunked and passed
    to _llm_extract_constraints so large sections are fully processed.
    """
    if not _MD_AVAILABLE:
        return _extract_prose(text)

    md = MarkdownIt()
    tokens = md.parse(text)
    nodes: list[dict] = []
    source_id = re.sub(r"[^a-zA-Z0-9_]", "_", source_name or "prose")

    current_heading: str = ""
    current_body_lines: list[str] = []
    section_bodies: list[tuple[str, str]] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            if current_heading:
                section_bodies.append((current_heading, "\n".join(current_body_lines)))
            current_body_lines = []
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                current_heading = tokens[i + 1].content.strip()
            else:
                current_heading = ""
        elif tok.type == "inline" and current_heading:
            if i > 0 and tokens[i - 1].type != "heading_open":
                current_body_lines.append(tok.content)
        i += 1

    if current_heading:
        section_bodies.append((current_heading, "\n".join(current_body_lines)))

    # If no headings found (flat prose), treat entire text as one constraint section
    if not section_bodies and text.strip():
        section_bodies = [("document", text)]

    for heading, body in section_bodies:
        heading_lower = heading.lower()
        node_id = f"{source_id}__{re.sub(r'[^a-zA-Z0-9_]', '_', heading)}"

        if heading_lower in _CONSTRAINT_HEADINGS and body.strip():
            # Chunk the section body — do not truncate
            constraints = _llm_extract_constraints_chunked(body, heading)
            for idx, c in enumerate(constraints):
                c_id = f"{node_id}__constraint_{idx}"
                c.setdefault("node_id", c_id)
                c.setdefault("label", "Execution_Constraint")
                c.setdefault("graph_type", "procedural")
                c.setdefault("content", c.get("enforced_by", heading)[:200])
                nodes.append(c)
        elif body.strip():
            content = body.strip().split("\n")[0][:200] or heading
            nodes.append({
                "frame_name": "Code_Concept",
                "node_id": node_id,
                "content": content,
                "label": "Concept",
                "graph_type": "semantic",
                "concept_type": "pattern",
                "source_file": source_name,
                "language": "prose",
            })

    return nodes


def _extract_prose(text: str) -> list[dict]:
    """Chunked constraint extraction for unstructured text with no heading structure."""
    if not text.strip():
        return []
    return _llm_extract_constraints_chunked(text, "document")


# ---------------------------------------------------------------------------
# LLM fallback functions
# ---------------------------------------------------------------------------

def _llm_summarize(node_id: str, context: str) -> str:
    """One-sentence content summary. Called when docstring is absent."""
    try:
        from orchestrator.llm_client import chat_completion
        result = chat_completion(
            [{
                "role": "user",
                "content": (
                    f"Write a single sentence (under 20 words) describing "
                    f"what this code does:\n\n{context[:500]}"
                ),
            }],
            temperature=0.0,
            max_tokens=60,
        )
        return str(result).strip().split("\n")[0][:200]
    except Exception as exc:
        logger.debug("_llm_summarize failed for %s: %s", node_id, exc)
        return ""


def _llm_classify_concept_type(node_id: str, class_info: dict) -> str:
    """Classify concept_type when AST heuristics return None."""
    valid = {
        "data_structure", "algorithm", "pattern",
        "interface", "protocol", "invariant",
    }
    try:
        from orchestrator.llm_client import chat_completion
        result = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the code structure. Reply with exactly one word from: "
                        "data_structure, algorithm, pattern, interface, protocol, invariant"
                    ),
                },
                {"role": "user", "content": str(class_info)[:600]},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        word = str(result).strip().lower().split()[0] if result else ""
        return word if word in valid else "pattern"
    except Exception as exc:
        logger.debug("_llm_classify_concept_type failed for %s: %s", node_id, exc)
        return "pattern"


def _llm_extract_constraints(text: str, section_name: str) -> list[dict]:
    """Extract ConstraintNode dicts from a single text chunk via LLM.

    This function processes one chunk. Call _llm_extract_constraints_chunked
    for prose sections longer than _CHUNK_SIZE characters.
    """
    import json as _json
    valid_types = {
        "memory_limit", "latency_limit", "api_contract",
        "type_contract", "security_requirement",
    }
    try:
        from orchestrator.llm_client import chat_completion
        result = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract constraints from the text as a JSON array. "
                        "Each item: {\"constraint_type\": one of "
                        "memory_limit|latency_limit|api_contract|type_contract|security_requirement, "
                        "\"enforced_by\": \"what enforces it\", "
                        "\"content\": \"one sentence\", "
                        "\"threshold\": \"limit value if stated or empty string\"}. "
                        "Return [] if no constraints found. JSON array only, no fences."
                    ),
                },
                {"role": "user", "content": f"Section: {section_name}\n\n{text}"},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw = str(result).strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rsplit("```", 1)[0].strip()
        items = _json.loads(raw) if raw else []
        out = []
        for item in items:
            ct = item.get("constraint_type", "")
            if ct not in valid_types:
                continue
            item["frame_name"] = "Execution_Constraint"
            item["graph_type"] = "procedural"
            item["label"] = "Execution_Constraint"
            item.setdefault("content", item.get("enforced_by", "")[:200])
            out.append(item)
        return out
    except Exception as exc:
        logger.debug("_llm_extract_constraints failed for %s: %s", section_name, exc)
        return []


def _llm_extract_constraints_chunked(text: str, section_name: str) -> list[dict]:
    """Extract ConstraintNode dicts from prose of any length.

    Splits text into overlapping chunks via _chunk_text, runs
    _llm_extract_constraints on each chunk, then deduplicates results
    by (constraint_type, enforced_by) key. Overlap ensures constraints
    that straddle a chunk boundary are captured.
    """
    chunks = _chunk_text(text)
    seen: set[tuple[str, str]] = set()
    results: list[dict] = []

    for chunk in chunks:
        for item in _llm_extract_constraints(chunk, section_name):
            key = (
                item.get("constraint_type", ""),
                item.get("enforced_by", ""),
            )
            if key not in seen and key[0]:
                seen.add(key)
                results.append(item)

    return results
