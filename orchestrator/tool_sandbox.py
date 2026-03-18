import subprocess
import re
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fleet-Sandbox")

@mcp.tool()
def execute_bash(code: str, timeout: int = 30):
    """Executes bash commands for filesystem observation (DRT/Frege)."""
    try:
        res = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=timeout)
        return res.stdout if res.stdout else res.stderr
    except subprocess.TimeoutExpired:
        return "Error: Bash execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def execute_python(code: str, timeout: int = 30):
    """Executes Python for task implementation."""
    try:
        res = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=timeout)
        return res.stdout if res.stdout else res.stderr
    except subprocess.TimeoutExpired:
        return "Error: Python execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information and save results to /workspace."""
    import subprocess
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        subprocess.run(["pip", "install", "-q", "duckduckgo-search"], capture_output=True)
        from duckduckgo_search import DDGS

    import os
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        output = []
        for r in results:
            output.append(f"## {r.get('title', '')}")
            output.append(f"{r.get('body', '')}")
            output.append(f"URL: {r.get('href', '')}")
            output.append("")

        result_text = "\n".join(output)

        # Write to workspace to trigger file delta for graph ingestion
        os.makedirs("/workspace", exist_ok=True)
        import hashlib
        h = hashlib.md5(query.encode()).hexdigest()[:6]
        path = f"/workspace/research_{h}.md"
        with open(path, "w") as f:
            f.write(f"# Research Results for: {query}\n\n")
            f.write(result_text)

        return result_text or "No search results."
    except Exception as e:
        return f"Web search error: {str(e)}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    import os
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    try:
        with open(path, "r") as f:
            content = f.read()
        res = content[:5000]
        if len(content) > 5000:
            res += f"\n... (truncated, total {len(content)} chars)"
        return res
    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def forward(code: str, language: str = "python", timeout: int = 30) -> dict:
    """
    FIXED: Added 'timeout' parameter to support WorkspaceState.capture calls
    in orchestrator/composition.py.
    """
    if language == "bash":
        return {"stdout": execute_bash(code, timeout=timeout)}
    return {"stdout": execute_python(code, timeout=timeout)}

def execute_code(raw_content: str) -> str:
    """Linguistic extraction of code blocks for tool routing."""
    match = re.search(r"```(python|bash)\n(.*?)\n```", raw_content, re.DOTALL)
    if not match: 
        return "No executable code found."
    lang, code = match.groups()
    if lang == "python":
        return execute_python(code)
    return execute_bash(code)
