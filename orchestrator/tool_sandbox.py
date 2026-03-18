"""Tool sandbox — execution primitives AND tool schema + dispatch.

All tool definitions and dispatch logic live here (separation of concerns).
The agent loop imports AVAILABLE_TOOLS and dispatch_tool from this module
rather than defining them inline.
"""

import json
import re
import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fleet-Sandbox")


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


def forward(code: str, language: str = "python", timeout: int = 30) -> dict:
    """Unified execution gateway — returns {"stdout": ...} dict.

    Used by WorkspaceState.capture() and by dispatch_tool().
    """
    if language == "bash":
        return {"stdout": execute_bash(code, timeout=timeout)}
    return {"stdout": execute_python(code, timeout=timeout)}


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
            query = arguments.get("query", "")
            code = f"""
try:
    from duckduckgo_search import DDGS
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "duckduckgo-search"], capture_output=True)
    from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text({repr(query)}, max_results=5))
    for r in results:
        print(f"## {{r.get('title', '')}}")
        print(f"{{r.get('body', '')}}")
        print(f"URL: {{r.get('href', '')}}")
        print()
"""
            result = forward(code=code, language="python", timeout=30)
            return result.get("stdout", "") or "No search results."

        elif name == "run_code":
            language = arguments.get("language", "python")
            code = arguments.get("code", "")
            result = forward(code=code, language=language, timeout=30)
            output = result.get("stdout", "")
            return output if output else "Code executed (no output)."

        elif name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
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

        elif name == "read_file":
            path = arguments.get("path", "")
            code = f"""
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

        elif name == "task_complete":
            return f"TASK_COMPLETE: {arguments.get('summary', 'Done')}"

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {str(e)}"
