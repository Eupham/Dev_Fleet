import subprocess
import re
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP for tool management
mcp = FastMCP("Jules-Sandbox")

@mcp.tool()
def execute_python(code: str) -> str:
    """Executes Python code in a secure sub-process."""
    try:
        result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=30)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def execute_bash(code: str) -> str:
    """Executes Bash commands. Required for WorkspaceState (sha256sum) captures."""
    try:
        result = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=30)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

def execute_code(raw_content: str, language: str = "python") -> str:
    """Helper to extract code blocks and route to the correct MCP tool."""
    code_match = re.search(rf"```{language}\n(.*?)\n```", raw_content, re.DOTALL)
    if not code_match:
        return f"No valid {language} code block found."
    
    if language == "python":
        return execute_python(code_match.group(1))
    return execute_bash(code_match.group(1))

def forward(code: str, language: str = "python") -> dict:
    """Generic interface required by CompositionLedger."""
    out_str = execute_code(f"```{language}\n{code}\n```", language=language)
    return {"stdout": out_str}
