"""Tool Sandbox — Ephemeral Modal Sandbox execution.

Code execution MUST occur in isolated Modal Sandboxes so the host
container is never at risk.  Each invocation creates a fresh sandbox,
runs the command, and captures stdout/stderr/exit-code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from smolagents import Tool


# ---------------------------------------------------------------------------
# Sandbox result schema
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Outcome of a single sandbox execution."""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        return self.exit_code == 0


# ---------------------------------------------------------------------------
# Tool Sandbox Class
# ---------------------------------------------------------------------------


class ModalSandboxTool(Tool):
    """A tool to execute bash or python code in a secure, ephemeral Modal sandbox."""

    name = "modal_sandbox_tool"
    description = (
        "Run code inside an ephemeral Modal Sandbox. "
        "Useful for safely executing untrusted Python or Bash code. "
        "Returns a dict with stdout, stderr, and exit_code."
    )
    inputs = {
        "code": {
            "type": "string",
            "description": "Source code or shell commands to execute."
        },
        "language": {
            "type": "string",
            "description": "The language to execute, either 'python' or 'bash'.",
            "nullable": True
        },
        "env": {
            "type": "object",
            "description": "Optional environment variables passed to the sandbox.",
            "nullable": True
        },
        "timeout": {
            "type": "integer",
            "description": "Maximum execution time in seconds.",
            "nullable": True
        }
    }
    output_type = "any" # Dictionary containing stdout, stderr, and exit_code

    def forward(
        self,
        code: str,
        language: str = "python",
        env: Optional[Dict[str, str]] = None,
        timeout: int = 120,
    ) -> dict:
        """Run *code* inside an ephemeral Modal Sandbox."""
        import modal  # lazy import — only needed at runtime on Modal

        # Sandbox image — stdlib + common task packages + network-capable tools
        _sandbox_image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("curl", "wget", "git")
            .pip_install(
                "pydantic>=2.5",
                "requests>=2.31",
                "httpx>=0.27",
                "beautifulsoup4>=4.12",
                "pygame-ce>=2.4",
                "matplotlib>=3.8",
                "numpy>=1.26",
            )
        )

        workspace_vol = modal.Volume.from_name(
            "dev_fleet-workspace", create_if_missing=True
        )

        app_ref = modal.App.lookup("dev_fleet", create_if_missing=True)

        sandbox = modal.Sandbox.create(
            app=app_ref,
            image=_sandbox_image,
            timeout=timeout,
            volumes={"/workspace": workspace_vol},
            network_access=True,
            **({"environment_variables": env} if env else {}),
        )

        try:
            if language == "python":
                proc = sandbox.exec("python", "-c", code)
            else:
                proc = sandbox.exec("bash", "-c", code)

            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            exit_code = proc.returncode
        except Exception as exc:
            stdout = ""
            stderr = str(exc)
            exit_code = 1
        finally:
            sandbox.terminate()

        # We return a dict as expected by the new Tool abstraction
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code if exit_code is not None else 1,
        }


# ---------------------------------------------------------------------------
# Public API Backwards Compatibility
# ---------------------------------------------------------------------------


def execute_in_sandbox(
    code: str,
    language: str = "python",
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> SandboxResult:
    """Run *code* inside an ephemeral Modal Sandbox.

    Parameters
    ----------
    code:
        Source code or shell commands to execute.
    language:
        ``"python"`` or ``"bash"``.
    env:
        Optional environment variables passed to the sandbox.
    timeout:
        Maximum execution time in seconds.

    Returns
    -------
    SandboxResult with stdout, stderr, and exit_code.
    """
    tool = ModalSandboxTool()
    res = tool.forward(code=code, language=language, env=env, timeout=timeout)
    return SandboxResult(
        stdout=res["stdout"],
        stderr=res["stderr"],
        exit_code=res["exit_code"]
    )
