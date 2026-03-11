#The script that will run in `dev_fleet` to pull documentation and save it to a Modal Volume as clean Markdown for later ingestion
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import datetime
from urllib.parse import urljoin, urlparse
import modal

# --- Modal App Setup ---
app = modal.App("dev_fleet_doc_ingestion")
volume = modal.Volume.from_name("dev_fleet_docs", create_if_missing=True)
jules_secret = modal.Secret.from_name("jules-api-key")

DOC_TARGETS = {
    "Modal": "https://modal.com/docs/guide",
    "vLLM": "https://docs.vllm.ai/en/latest/",
    "FastAPI": "https://fastapi.tiangolo.com/",
    "DSPy": "https://dspy-docs.vercel.app/docs/intro",
    "LiteLLM": "https://docs.litellm.ai/docs/",
    "JulesAPI": "https://developers.google.com/jules/api",
    "JulesREST": "https://developers.google.com/jules/api/reference/rest"
}

def get_links(base_url: str) -> list:
    """Crawls the base URL to find relevant documentation links."""
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Failed to fetch {base_url}: {e}")
        return []

    domain = urlparse(base_url).netloc
    path = urlparse(base_url).path
    links = set([base_url])
    
    for a_tag in soup.find_all('a', href=True):
        full_url = urljoin(base_url, a_tag.get('href', ''))
        clean_url = full_url.split('#')[0] # Remove anchor tags
        if urlparse(clean_url).netloc == domain and path in urlparse(clean_url).path:
            links.add(clean_url)
            
    return sorted(list(links))

def extract_to_markdown(url: str) -> tuple:
    """Extracts main content and converts it to clean Markdown."""
    try:
        # Be polite to the servers
        time.sleep(0.5)

        response = requests.get(url, timeout=10)
        status = response.status_code
        if status != 200:
             print(f"[{status}] Failed to fetch: {url}")
             return "Error", url, f"HTTP Error {status} on {url}"

        print(f"[{status}] Successfully fetched: {url}")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Target the main content area to avoid navbars and sidebars
        content = soup.find('main') or soup.find('article') or soup.find('div', role='main')
        if not content:
            content = soup.body
            
        title = soup.find('title').get_text(strip=True) if soup.find('title') else url
        
        # Convert HTML to Markdown, stripping out images and preserving code blocks
        raw_markdown = md(str(content), heading_style="ATX", strip=['img', 'script', 'style'])
        return title, url, raw_markdown
    except Exception as e:
        print(f"[ERROR] Exception on {url}: {e}")
        return "Error", url, f"Failed to extract {url}: {e}"

def process_target(name: str, base_url: str) -> tuple:
    """Processes a single documentation target and its links."""
    print(f"Rebuilding {name} knowledge base...")
    links = get_links(base_url)
    print(f"[{name}] Found {len(links)} links to process.")

    pages_data = []

    # Process links in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        # submit all links to the executor
        future_to_url = {executor.submit(extract_to_markdown, url): url for url in links}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                title, processed_url, markdown_content = future.result()
                pages_data.append((title, processed_url, markdown_content))
            except Exception as exc:
                print(f"[{name}] {url} generated an exception: {exc}")

    return name, base_url, pages_data

@app.function(
    schedule=modal.Cron("0 0 1 * *"), 
    volumes={"/docs_volume": volume},
    image=modal.Image.debian_slim().pip_install("requests", "beautifulsoup4", "markdownify"),
    timeout=1800, # 30 minutes instead of default 5 minutes
    secrets=[jules_secret, modal.Secret.from_name("jules-session-id")]
)
def rebuild_knowledge_base():
    """Crawls targets and compiles them into master Markdown files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Process documentation targets in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(DOC_TARGETS)) as target_executor:
        future_to_target = {target_executor.submit(process_target, name, url): name for name, url in DOC_TARGETS.items()}
        for future in as_completed(future_to_target):
            name = future_to_target[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"[MASTER] Target {name} generated an exception: {exc}")

    # Save the processed data to markdown files sequentially
    for name, base_url, pages_data in results:
        master_md = f"# {name} Master Documentation\n"
        master_md += f"**Last Updated:** {timestamp}\n\n"
        master_md += "## Index\n"
        
        # Ensure consistent ordering
        pages_data.sort(key=lambda x: x[1])

        for i, (title, url, _) in enumerate(pages_data):
            anchor = title.lower().replace(" ", "-").replace("/", "")
            master_md += f"{i+1}. [{title}](#{anchor})\n"
            
        master_md += "\n---\n\n"
        
        for title, url, markdown_content in pages_data:
            master_md += f"## {title}\n"
            master_md += f"*Source: {url}*\n\n"
            master_md += f"{markdown_content}\n\n"
            master_md += "---\n\n"
            
        file_path = f"/docs_volume/{name.lower()}_master.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(master_md)
            
        print(f"✅ Saved {name} to {file_path}")
        
    volume.commit()

    # Notify Jules API that ingestion is complete
    print(f"JULES_API_KEY environment variable is present: {'JULES_API_KEY' in os.environ}")
    api_key = os.environ.get("JULES_API_KEY", "")
    session_id = os.environ.get("JULES_SESSION_ID", "")

    if not api_key:
        print("Error: Missing JULES_API_KEY")
        return

    if not session_id:
        print("Error: Missing JULES_SESSION_ID")
        return

    print(f"Ingestion complete. API Key length: {len(api_key)}")

    try:
        import requests
        headers = {
            "X-Goog-Api-Key": api_key,
            "Content-Type": "application/json"
        }

        # Send message to the Jules API
        # Handle cases where session_id might just be the ID or the full 'sessions/ID' string
        if not session_id.startswith("sessions/"):
            session_id = f"sessions/{session_id}"

        url = f"https://jules.googleapis.com/v1alpha/{session_id}:sendMessage"
        response = requests.post(
            url,
            headers=headers,
            json={"prompt": "Knowledge base ingestion finished successfully."},
            timeout=10
        )
        print(f"Notified Jules API. Status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Failed to notify Jules API: {e}")
