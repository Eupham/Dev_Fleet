import subprocess
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP for tool management
mcp = FastMCP("Jules-Sandbox")

@mcp.tool()
def execute_python(code: str) -> str:
    """Executes Python code in a secure sub-process and returns stdout/stderr."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out (30s limit)."
    except Exception as e:
        return f"Error: {str(e)}"

def execute_code(raw_content: str) -> str:
    """Helper to extract code blocks and route to the MCP tool."""
    import re
    code_match = re.search(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
    if not code_match:
        return "No valid python code block found."
    return execute_python(code_match.group(1))
