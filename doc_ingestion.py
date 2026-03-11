import os
import time
import concurrent.futures
import requests_cache
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import datetime
from urllib.parse import urljoin, urlparse
import modal

# --- Modal App Setup ---
app = modal.App("dev_fleet_doc_ingestion")
volume = modal.Volume.from_name("dev_fleet_docs", create_if_missing=True)

# --- Phase 3 AI-Native Roadmap Notes ---
"""
FUTURE STATE (Phase 3 Knowledge Graph Pivot):
This script uses `requests-cache` to maintain static Markdown files for bootstrapping.
In Phase 3, this script will be upgraded to an AI-native ingestion pipeline:
1. Chunking: Text will be semantically chunked.
2. Embedding: Chunks will be embedded using https://huggingface.co/collections/Qwen/qwen3-embedding and stored in Qdrant.
3. Reranking: https://huggingface.co/collections/Qwen/qwen3-reranker will evaluate chunk relevance.
4. Knowledge Graph: Highly ranked chunks will become reference nodes within the NetworkX graph for the fleet_fix agent.
"""

# Set up persistent SQLite cache on the Modal Volume (expires after 24 hours)
cache_path = '/docs_volume/http_cache'
session = requests_cache.CachedSession(
    cache_name=cache_path,
    backend='sqlite',
    expire_after=86400
)

DOC_TARGETS = {
    "Modal": "https://modal.com/docs/guide",
    "vLLM": "https://docs.vllm.ai/en/latest/",
    "FastAPI": "https://fastapi.tiangolo.com/",
    "DSPy": "https://dspy-docs.vercel.app/docs/intro",
    "LiteLLM": "https://docs.litellm.ai/docs/",
    "JulesAPI": "https://developers.google.com/jules/api"
}

def sanitize_url(url: str) -> str:
    """Strips fragments and query parameters to prevent duplicate page fetches."""
    clean = url.split('#')[0].split('?')[0]
    return clean.rstrip('/')

def get_links(base_url: str) -> list:
    try:
        response = session.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Failed to fetch base URL {base_url}: {e}")
        return []

    domain = urlparse(base_url).netloc
    path = urlparse(base_url).path
    links = set([sanitize_url(base_url)])

    for a_tag in soup.find_all('a', href=True):
        full_url = urljoin(base_url, a_tag.get('href', ''))
        clean_url = sanitize_url(full_url)
        if urlparse(clean_url).netloc == domain and path in urlparse(clean_url).path:
            links.add(clean_url)

    return sorted(list(links))

def extract_to_markdown(url: str) -> tuple:
    time.sleep(0.1) # Cache absorbs most hits, but slight delay ensures politeness on misses
    print(f"Fetching: {url}")
    try:
        response = session.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body
        title = soup.find('title').get_text(strip=True) if soup.find('title') else url
        raw_markdown = md(str(content), heading_style="ATX", strip=['img', 'script', 'style'])
        return title, url, raw_markdown
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return "Error", url, f"Failed to extract: {e}"

@app.function(
    schedule=modal.Cron("0 0 1 * *"),
    volumes={"/docs_volume": volume},
    image=modal.Image.debian_slim().pip_install("requests-cache", "beautifulsoup4", "markdownify"),
    secrets=[modal.Secret.from_name("jules-api-keys", required_keys=["JULES_API_KEY", "JULES_SESSION_ID"])],
    timeout=3600
)
def rebuild_knowledge_base():
    """Crawls targets in parallel and compiles them into master Markdown files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    for name, base_url in DOC_TARGETS.items():
        print(f"\n--- Rebuilding {name} ---")
        links = get_links(base_url)
        print(f"Found {len(links)} unique pages for {name}.")

        master_md = f"# {name} Master Documentation\n**Last Updated:** {timestamp}\n\n## Index\n"
        pages_data = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(extract_to_markdown, links))

        for i, (title, url, markdown_content) in enumerate(results):
            pages_data.append((title, url, markdown_content))
            anchor = title.lower().replace(" ", "-").replace("/", "")
            master_md += f"{i+1}. [{title}](#{anchor})\n"

        master_md += "\n---\n\n"

        for title, url, markdown_content in pages_data:
            master_md += f"## {title}\n*Source: {url}*\n\n{markdown_content}\n\n---\n\n"

        file_path = f"/docs_volume/{name.lower()}_master.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(master_md)

        print(f"✅ Saved {name} to {file_path}")

    volume.commit()
    print("\n✅ Knowledge base built. Caching database saved to volume.")

    # --- Jules API Async Notification Block ---
    import requests # Standard requests here to bypass the cache for the POST ping
    api_key = os.environ.get("JULES_API_KEY")
    session_id = os.environ.get("JULES_SESSION_ID")

    if api_key and session_id:
        print(f"\nNotifying Jules session: {session_id}")
        url = f"https://jules.googleapis.com/v1alpha/{session_id}:sendMessage"
        headers = {"X-Goog-Api-Key": api_key, "Content-Type": "application/json"}
        data = {"prompt": "SYSTEM AUTOMATION: The doc_ingestion task has completed successfully. All knowledge base files are now available in the Modal Volume. Await further user instructions or proceed to the next available step in Phase 1 of master_instructions.md."}

        try:
            ping = requests.post(url, headers=headers, json=data)
            ping.raise_for_status()
            print("Successfully notified Jules.")
        except Exception as e:
            print(f"Failed to notify Jules: {e}")

@app.local_entrypoint()
def main():
    print("Triggering remote knowledge base rebuild...")
    rebuild_knowledge_base.remote()