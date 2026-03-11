#The script that will run in `dev_fleet` to pull documentation and save it to a Modal Volume as clean Markdown for later ingestion
import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import datetime
from urllib.parse import urljoin, urlparse
import modal

# --- Modal App Setup ---
app = modal.App("dev_fleet_doc_ingestion")
volume = modal.Volume.from_name("dev_fleet_docs", create_if_missing=True)

DOC_TARGETS = {
    "Modal": "https://modal.com/docs/guide",
    "vLLM": "https://docs.vllm.ai/en/latest/",
    "FastAPI": "https://fastapi.tiangolo.com/",
    "DSPy": "https://dspy-docs.vercel.app/docs/intro",
    "LiteLLM": "https://docs.litellm.ai/docs/"
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
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Target the main content area to avoid navbars and sidebars
        content = soup.find('main') or soup.find('article') or soup.find('div', role='main')
        if not content:
            content = soup.body
            
        title = soup.find('title').get_text(strip=True) if soup.find('title') else url
        
        # Convert HTML to Markdown, stripping out images and preserving code blocks
        raw_markdown = md(str(content), heading_style="ATX", strip=['img', 'script', 'style'])
        return title, raw_markdown
    except Exception as e:
        return "Error", f"Failed to extract {url}: {e}"

@app.function(
    schedule=modal.Cron("0 0 1 * *"), 
    volumes={"/docs_volume": volume},
    image=modal.Image.debian_slim().pip_install("requests", "beautifulsoup4", "markdownify")
)
def rebuild_knowledge_base():
    """Crawls targets and compiles them into master Markdown files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    for name, base_url in DOC_TARGETS.items():
        print(f"Rebuilding {name} knowledge base...")
        links = get_links(base_url)
        
        master_md = f"# {name} Master Documentation\n"
        master_md += f"**Last Updated:** {timestamp}\n\n"
        master_md += "## Index\n"
        
        pages_data = []
        for i, url in enumerate(links):
            title, markdown_content = extract_to_markdown(url)
            pages_data.append((title, url, markdown_content))
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
