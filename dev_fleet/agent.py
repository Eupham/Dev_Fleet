import modal
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os

app = modal.App("dev-fleet-agent")
web_app = FastAPI()
volume = modal.Volume.from_name("alt_git_repos", create_if_missing=True)

class TaskPrompt(BaseModel):
    repo_name: str
    prompt: str

class IssueWebhook(BaseModel):
    issue_id: str
    traceback: str
    iteration: int

# Container image with dependencies
image = modal.Image.debian_slim().apt_install("git").pip_install(
    "dspy-ai",
    "aider-chat",
    "vllm",
    "transformers",
    "fastapi[standard]"
)

# Background task function to process fixes
@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=600)
def decompose_and_execute(repo_name: str, task_description: str):
    print(f"Starting agent pipeline for repo {repo_name}...")
    print(f"Task: {task_description}")

    # In a full implementation, this is where vLLM, DSPy, and Aider would be invoked
    # e.g.:
    # 1. Start vLLM inference engine pointing to a local/cached Qwen model
    # 2. Use DSPy to reason about task decomposition
    # 3. Call Aider CLI headlessly in /data/repos/{repo_name} to execute changes

    repo_path = f"/data/repos/{repo_name}"

    print(f"Simulating decomposition using vLLM...")
    print(f"Simulating dspy reasoning...")
    print(f"Executing changes headlessly with Aider in {repo_path}...")

    # Simulate work
    if os.path.exists(repo_path):
        # We would actually edit code and commit here
        print(f"Agent work completed successfully on {repo_name}")
    else:
        print(f"Repo path {repo_path} does not exist.")

@web_app.post("/agent/prompt")
async def process_user_prompt(prompt: TaskPrompt, background_tasks: BackgroundTasks):
    print(f"Received user prompt for repo {prompt.repo_name}")
    # Run the heavy agent task in the background via Modal's remote execution
    decompose_and_execute.spawn(prompt.repo_name, prompt.prompt)
    return {"status": "success", "message": "Agent task spawned"}

@web_app.post("/agent/webhook/issue")
async def process_issue_webhook(webhook: IssueWebhook, background_tasks: BackgroundTasks):
    print(f"Received issue webhook for {webhook.issue_id}, iteration {webhook.iteration}")

    # Extract repo context if it exists (assuming it might be part of the traceback or mapped)
    # For now, default to "test_repo" for the demo
    repo_name = "test_repo"
    task_description = f"Fix error from issue {webhook.issue_id}. Traceback: {webhook.traceback}"

    # Run agent in background
    decompose_and_execute.spawn(repo_name, task_description)
    return {"status": "success", "message": "Self-healing task spawned"}

@app.function(image=image)
@modal.asgi_app()
def agent_app():
    return web_app
