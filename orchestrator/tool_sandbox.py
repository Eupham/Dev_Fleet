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
