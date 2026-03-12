import modal
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fleet_app import app

# Create a lightweight CPU image for the web UI
web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi", "uvicorn", "jinja2", "python-multipart")

web_app = FastAPI()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dev Fleet UI</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 100px; margin-bottom: 10px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #ddd; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Dev Fleet Orchestrator</h1>
    <form method="post" action="/">
        <textarea name="prompt" placeholder="Enter your prompt here..." required>{{ prompt }}</textarea>
        <button type="submit">Run Orchestrator</button>
    </form>

    {% if result %}
    <div class="result">
        <h2>Result:</h2>
        <pre>{{ result }}</pre>
    </div>
    {% endif %}
</body>
</html>
"""

from jinja2 import Template

@web_app.get("/", response_class=HTMLResponse)
async def home():
    template = Template(HTML_TEMPLATE)
    return template.render(prompt="", result="")

@web_app.post("/", response_class=HTMLResponse)
async def run(prompt: str = Form(...)):
    from orchestrator.core_app import run_agent

    # Call the orchestrator remotely
    result = run_agent.remote(prompt)

    import json
    formatted_result = json.dumps(result, indent=2)

    template = Template(HTML_TEMPLATE)
    return template.render(prompt=prompt, result=formatted_result)

@app.function(image=web_image)
@modal.asgi_app()
def ui():
    return web_app
