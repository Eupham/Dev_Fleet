# orchestrator/document_fetcher.py
"""URL fetching and crawling for domain graph construction.

Entry points:
  fetch_url(url)                   → FetchedDocument
  crawl_urls(urls, depth, domain)  → list[FetchedDocument]

Uses trafilatura.fetch_url for HTTP fetching (handles user-agent,
rate limiting, redirects). PDF bytes are passed to _extract_pdf.

Crawling is bounded by:
  depth:  max link-follow depth from seed URLs (default 1)
  domain: if set, only follow links on the same domain
  max_pages: hard cap (default 50) to prevent runaway crawls
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("dev_fleet.document_fetcher")

_MAX_PAGES_DEFAULT = 50
_SEEN_URLS: set[str] = set()   # module-level dedup across calls in one session


@dataclass
class FetchedDocument:
    url: str
    content: str          # raw HTML or text
    content_type: str     # "html" | "pdf" | "text"
    title: str = ""
    outbound_links: list[str] = field(default_factory=list)
    error: str = ""


def fetch_url(url: str) -> FetchedDocument:
    """Fetch a single URL. Returns a FetchedDocument.

    Uses trafilatura.fetch_response for HTML/text.
    Detects PDFs by Content-Type or URL suffix.
    """
    try:
        import trafilatura
    except ImportError:
        return FetchedDocument(url=url, content="", content_type="text",
                               error="trafilatura not installed")

    try:
        response = trafilatura.fetch_response(url)
        if response is None:
            return FetchedDocument(url=url, content="", content_type="text",
                                   error="fetch returned None")

        content_type_header = getattr(response, "headers", {}).get(
            "Content-Type", ""
        ).lower()

        # PDF detection
        if "pdf" in content_type_header or url.lower().endswith(".pdf"):
            raw_bytes = response.data if hasattr(response, "data") else b""
            return FetchedDocument(
                url=url,
                content=raw_bytes,   # type: ignore[arg-type]
                content_type="pdf",
                title=_title_from_url(url),
            )

        # HTML extraction via trafilatura bare_extraction
        html = response.data.decode("utf-8", errors="replace") if hasattr(response, "data") else ""
        if not html:
            return FetchedDocument(url=url, content="", content_type="text",
                                   error="empty response")

        # Extract metadata (title, outbound links)
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = meta.title if meta and meta.title else _title_from_url(url)

        # Extract outbound links via bare_extraction
        doc = trafilatura.bare_extraction(
            html,
            url=url,
            include_links=True,
            include_formatting=True,
            deduplicate=True,
            with_metadata=True,
            favor_recall=True,
        )
        links: list[str] = []
        if doc:
            xml_str = trafilatura.extract(
                html, output_format="xml", include_links=True, favor_recall=True
            ) or ""
            # Extract href values from <ref target="..."> in trafilatura XML
            links = re.findall(r'<ref[^>]+target="(https?://[^"]+)"', xml_str)[:100]

        return FetchedDocument(
            url=url,
            content=html,
            content_type="html",
            title=title,
            outbound_links=links,
        )

    except Exception as exc:
        logger.warning("fetch_url failed for %s: %s", url, exc)
        return FetchedDocument(url=url, content="", content_type="text",
                               error=str(exc))


def crawl_urls(
    seed_urls: list[str],
    depth: int = 1,
    same_domain_only: bool = True,
    max_pages: int = _MAX_PAGES_DEFAULT,
) -> list[FetchedDocument]:
    """Fetch seed URLs and optionally follow outbound links.

    Args:
        seed_urls:        Starting URLs.
        depth:            0 = only seeds. 1 = seeds + their links. etc.
        same_domain_only: If True, only follow links on same domain as seed.
        max_pages:        Hard cap on total pages fetched.

    Returns:
        List of FetchedDocument objects for all fetched pages.
    """
    seed_domains = {urlparse(u).netloc for u in seed_urls}
    queue: list[tuple[str, int]] = [(u, 0) for u in seed_urls]
    visited: set[str] = set()
    results: list[FetchedDocument] = []

    while queue and len(results) < max_pages:
        url, current_depth = queue.pop(0)
        if url in visited or url in _SEEN_URLS:
            continue
        visited.add(url)
        _SEEN_URLS.add(url)

        doc = fetch_url(url)
        if doc.error:
            logger.debug("Skipping %s: %s", url, doc.error)
            continue
        results.append(doc)

        if current_depth < depth and doc.outbound_links:
            for link in doc.outbound_links:
                if link in visited:
                    continue
                if same_domain_only and urlparse(link).netloc not in seed_domains:
                    continue
                queue.append((link, current_depth + 1))

    logger.info("crawl_urls: fetched %d pages from %d seeds (depth=%d)",
                len(results), len(seed_urls), depth)
    return results


def _title_from_url(url: str) -> str:
    """Extract a human-readable title from a URL path."""
    path = urlparse(url).path.rstrip("/")
    if path:
        last_segment = path.split("/")[-1]
        return re.sub(r"[_-]", " ", last_segment).replace(".pdf", "").strip()
    return urlparse(url).netloc
