import modal
import requests_cache
from bs4 import BeautifulSoup
import markdownify
import os

app = modal.App("dev-fleet-docs")
volume = modal.Volume.from_name("dev_fleet_docs", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "requests-cache",
    "beautifulsoup4",
    "markdownify"
)

@app.function(image=image, volumes={"/data": volume})
def ingest_docs():
    """
    Ingests documentation for frameworks.

    In Phase 3, this script will be upgraded to chunk and embed the text
    using Qwen3 embeddings and a NetworkX knowledge graph.
    """
    # Setup requests cache using sqlite backend stored in the volume
    session = requests_cache.CachedSession('/data/docs_cache', backend='sqlite')

    # Example docs to fetch
    docs_to_fetch = {
        "fastapi": "https://fastapi.tiangolo.com/",
        "vllm": "https://docs.vllm.ai/en/latest/",
        "dspy": "https://dspy-docs.vercel.app/"
    }

    for name, url in docs_to_fetch.items():
        print(f"Fetching docs for {name} from {url}")
        try:
            response = session.get(url)
            response.raise_for_status()

            # Parse HTML and convert to Markdown
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main content if possible, or just the whole body
            content = soup.body if soup.body else soup
            markdown_content = markdownify.markdownify(str(content), heading_style="ATX")

            # Save to volume
            filepath = f"/data/{name}_docs.md"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Successfully saved {name} docs to {filepath}")

        except Exception as e:
            print(f"Error fetching or processing docs for {name}: {e}")

    # Commit changes to the volume to persist them
    volume.commit()
