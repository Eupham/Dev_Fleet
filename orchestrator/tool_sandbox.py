"""Tool sandbox — execution primitives AND tool schema + dispatch.

All tool definitions and dispatch logic live here (separation of concerns).
The agent loop imports AVAILABLE_TOOLS and dispatch_tool from this module
rather than defining them inline.
"""

import json
import re
import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fleet-Sandbox", host="0.0.0.0")


# ---------------------------------------------------------------------------
# Low-level executors (used directly and via dispatch)
# ---------------------------------------------------------------------------

@mcp.tool()
def execute_bash(code: str, timeout: int = 30) -> str:
    """Execute bash commands in the sandbox (filesystem observation, installs)."""
    try:
        res = subprocess.run(
            ["bash", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.stdout if res.stdout else res.stderr
    except subprocess.TimeoutExpired:
        return "Error: Bash execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def execute_python(code: str, timeout: int = 30) -> str:
    """Execute Python code in the sandbox (task implementation)."""
    try:
        res = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.stdout if res.stdout else res.stderr
    except subprocess.TimeoutExpired:
        return "Error: Python execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def playwright_screenshot(url: str, output_path: str = "/workspace/screenshot.png", timeout: int = 30) -> str:
    """Take a full-page screenshot of a URL using Playwright.
    Returns the path to the saved screenshot or an error message.
    """
    code = f"""
import sys
import asyncio
from playwright.async_api import async_playwright
async def run():
    url = sys.argv[1]
    output_path = sys.argv[2]
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=15000)
            await page.screenshot(path=output_path, full_page=True)
            print(f"Screenshot saved to {{output_path}}")
        except Exception as e:
            print(f"Error navigating or taking screenshot: {{e}}")
        finally:
            await browser.close()
asyncio.run(run())
"""
    try:
        res = subprocess.run(
            ["python3", "-c", code, url, output_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.stdout if res.stdout else res.stderr
    except subprocess.TimeoutExpired:
        return "Error: Playwright execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def playwright_extract_text(url: str, timeout: int = 30) -> str:
    """Extract visible text from a URL using Playwright.
    Useful for reading websites that require JavaScript to render.
    """
    code = f"""
import sys
import asyncio
from playwright.async_api import async_playwright
async def run():
    url = sys.argv[1]
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=15000)
            # Wait for network idle or main content to load
            await page.wait_for_load_state("networkidle", timeout=10000)
            text = await page.evaluate("document.body.innerText")
            print(text[:10000]) # cap output length slightly
        except Exception as e:
            print(f"Error navigating or extracting text: {{e}}")
        finally:
            await browser.close()
asyncio.run(run())
"""
    try:
        res = subprocess.run(
            ["python3", "-c", code, url],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.stdout if res.stdout else res.stderr
    except subprocess.TimeoutExpired:
        return "Error: Playwright execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"


def forward(code: str, language: str = "python", timeout: int = 30) -> dict:
    """Unified execution gateway — returns {"stdout": ...} dict.

    Used by WorkspaceState.capture() and by dispatch_tool().
    """
    if language == "bash":
        return {"stdout": execute_bash(code, timeout=timeout)}
    return {"stdout": execute_python(code, timeout=timeout)}


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information. Use for research tasks, finding documentation, discovering libraries."""
    code = f"""
from ddgs import DDGS
results = []
try:
    with DDGS() as ddgs:
        results = list(ddgs.text({repr(query)}, max_results=6))
except Exception as e:
    print(f"Search error: {{e}}")

if results:
    for r in results:
        print(f"## {{r.get('title', '')}}")
        print(r.get('body', '') or r.get('snippet', ''))
        print(f"URL: {{r.get('href', '') or r.get('url', '')}}")
        print()
else:
    print("No results found for: {repr(query)}")
"""
    result = forward(code=code, language="python", timeout=30)
    output = result.get("stdout", "").strip()
    return output if output else f"No search results for: {query}"


@mcp.tool()
def run_code(language: str, code: str) -> str:
    """Execute Python or bash code in the sandbox. Returns stdout/stderr."""
    result = forward(code=code, language=language, timeout=30)
    output = result.get("stdout", "")
    return output if output else "Code executed (no output)."


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    code = f"""
import os, json
path = {json.dumps(path)}
content = {json.dumps(content)}
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as f:
    f.write(content)
print(f"Written {{len(content)}} chars to {{path}}")
"""
    result = forward(code=code, language="python", timeout=10)
    return result.get("stdout", "") or "File written."


@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    code = f"""
import json
path = {json.dumps(path)}
try:
    with open(path, "r") as f:
        content = f.read()
    print(content[:5000])
    if len(content) > 5000:
        print(f"\\n... (truncated, total {{len(content)}} chars)")
except FileNotFoundError:
    print(f"File not found: {{path}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
    result = forward(code=code, language="python", timeout=10)
    return result.get("stdout", "") or "File read failed."


@mcp.tool()
def task_complete(summary: str) -> str:
    """Signal that the current task is finished. MUST be called when done."""
    return f"TASK_COMPLETE: {summary}"


def execute_code(raw_content: str) -> str:
    """Legacy: extract and run a fenced code block from a raw model response."""
    match = re.search(r"```(python|bash)\n(.*?)\n```", raw_content, re.DOTALL)
    if not match:
        return "No executable code found."
    lang, code = match.groups()
    if lang == "python":
        return execute_python(code)
    return execute_bash(code)


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenAI-compatible, forwarded to llama-server)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information. Use for research tasks, "
                "finding documentation, discovering libraries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python or bash code in the sandbox. Returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "bash"]},
                    "code": {"type": "string", "description": "Code to execute"},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to /workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to /workspace",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Signal that the current task is finished. MUST be called when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "playwright_screenshot",
            "description": "Take a full-page screenshot of a URL using Playwright.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to capture",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save screenshot (default /workspace/screenshot.png)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "playwright_extract_text",
            "description": "Extract visible text from a URL using Playwright. Useful for reading JS-rendered websites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to extract text from",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch — routes LLM tool_calls to sandbox executors
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a named tool call and return its string output.

    This is the single authoritative dispatcher.  The agent loop must
    import and call this function — it must NOT duplicate dispatch logic.
    """
    try:
        if name == "web_search":
            return web_search(arguments.get("query", ""))

        elif name == "run_code":
            return run_code(arguments.get("language", "python"), arguments.get("code", ""))

        elif name == "write_file":
            return write_file(arguments.get("path", ""), arguments.get("content", ""))

        elif name == "read_file":
            return read_file(arguments.get("path", ""))

        elif name == "task_complete":
            return task_complete(arguments.get("summary", "Done"))

        elif name == "playwright_screenshot":
            url = arguments.get("url", "")
            output_path = arguments.get("output_path", "/workspace/screenshot.png")
            return playwright_screenshot(url, output_path)

        elif name == "playwright_extract_text":
            return playwright_extract_text(arguments.get("url", ""))

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {str(e)}"
