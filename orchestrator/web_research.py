"""Web Research Node — online research via sandbox execution.

The agent has internet access through the Modal Sandbox (curl, wget, requests, httpx
are all available). This node generates and executes a research script, then returns
parsed results as context for the decomposition step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.agent_loop import AgentState

logger = logging.getLogger("dev_fleet.web_research")

_RESEARCH_SYSTEM = """You are a web research assistant. Given a topic or question, write a Python script
that:
1. Uses requests or httpx to fetch relevant web pages or APIs
2. Parses the results (use BeautifulSoup for HTML, json for APIs)
3. Prints a structured summary of findings to stdout

Rules:
- Use only: requests, httpx, beautifulsoup4, json (all available in the sandbox)
- Handle errors gracefully with try/except
- Print findings in a clear, structured format
- Keep the script under 100 lines
- Do NOT use selenium or playwright (not available)

Output ONLY the Python script, no explanations."""


def research_node(state: "AgentState") -> dict:
    """Execute web research and return findings as context.

    Generates a Python research script, runs it in the sandbox, and
    returns the stdout as research context for the decompose step.
    """
    logger.info("Research node: conducting web research...")
    from orchestrator.llm_client import chat_completion
    from orchestrator.tool_sandbox import ModalSandboxTool
    import re

    user_prompt = state["user_prompt"]

    # Generate research script
    messages = [
        {"role": "system", "content": _RESEARCH_SYSTEM},
        {"role": "user", "content": f"Research topic: {user_prompt}"},
    ]

    research_results = "(no research results)"
    try:
        script_text = chat_completion(messages, temperature=0.3, max_tokens=2048)

        # Extract code block if wrapped in markdown
        match = re.search(r"```python\n(.*?)\n```", script_text, re.DOTALL)
        if match:
            script = match.group(1)
        else:
            script = script_text.strip()

        # Execute in sandbox
        tool = ModalSandboxTool()
        result = tool.forward(code=script, language="python", timeout=120)

        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()
        exit_code = result.get("exit_code", 1)

        if stdout:
            research_results = stdout[:4000]
            logger.info("Research completed: %d bytes of results", len(stdout))
        elif stderr:
            research_results = f"[Research error] {stderr[:1000]}"
            logger.warning("Research script produced errors: %s", stderr[:200])
        else:
            research_results = "[Research script produced no output]"

    except Exception as exc:
        logger.warning("Research node failed (%s) — proceeding without results.", exc)
        research_results = f"[Research failed: {exc}]"

    research_context = f"=== WEB RESEARCH RESULTS ===\n{research_results}\n=== END RESEARCH ==="

    return {
        "codebase_context": (
            (state.get("codebase_context", "") + "\n\n" + research_context).strip()
        ),
        "messages": [f"[RESEARCH] Completed web research ({len(research_results)} chars of findings)."],
    }
