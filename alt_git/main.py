import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import git
import os
import httpx

app = modal.App("alt-git-ledger")
volume = modal.Volume.from_name("alt_git_repos", create_if_missing=True)
web_app = FastAPI()

# Configuration
REPOS_DIR = "/data/repos"
DEV_FLEET_WEBHOOK_URL = os.environ.get("DEV_FLEET_WEBHOOK_URL", "https://bettergrads--dev-fleet-agent-agent-app.modal.run/agent/webhook/issue")

class CrashReport(BaseModel):
    issue_id: str
    traceback: str
    iteration: int

class RepoCreate(BaseModel):
    repo_name: str

@web_app.post("/repo")
async def create_repo(repo: RepoCreate):
    repo_path = os.path.join(REPOS_DIR, repo.repo_name)
    if os.path.exists(repo_path):
        raise HTTPException(status_code=400, detail="Repo already exists")

    os.makedirs(repo_path, exist_ok=True)
    git.Repo.init(repo_path)
    # Commit changes to volume
    await volume.commit.aio()
    return {"status": "success", "message": f"Repo {repo.repo_name} created"}

@web_app.get("/repo/{repo_name}/diff")
async def get_repo_diff(repo_name: str):
    repo_path = os.path.join(REPOS_DIR, repo_name)
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail="Repo not found")

    try:
        repo = git.Repo(repo_path)

        try:
            diff = repo.git.diff('HEAD')
            return {"status": "success", "diff": diff}
        except Exception as e:
            return {"status": "success", "diff": ""}

    except Exception as e:
        return {"status": "success", "diff": ""}

@web_app.post("/repo/{repo_name}/rollback")
async def rollback_repo(repo_name: str):
    repo_path = os.path.join(REPOS_DIR, repo_name)
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail="Repo not found")

    try:
        repo = git.Repo(repo_path)
        try:
            repo.git.reset('--hard', 'HEAD~1')
            await volume.commit.aio()
            return {"status": "success", "message": "Rolled back to HEAD~1"}
        except git.exc.GitCommandError:
            # Cannot rollback, possibly no commits or not enough commits
            return {"status": "error", "message": "Cannot rollback, not enough commits"}
    except Exception as e:
        return {"status": "error", "message": "Cannot rollback, not enough commits"}

@web_app.post("/telemetry/crash")
async def receive_crash_report(report: CrashReport):
    print(f"Received crash report for issue {report.issue_id}, iteration {report.iteration}")
    print(f"Traceback: {report.traceback}")

    # Trigger dev_fleet webhook
    try:
        async with httpx.AsyncClient() as client:
            # Use dict() for compatibility with older pydantic versions, fallback to model_dump()
            report_dict = report.dict() if hasattr(report, 'dict') else report.model_dump()
            response = await client.post(
                DEV_FLEET_WEBHOOK_URL,
                json=report_dict,
                timeout=10.0
            )
            response.raise_for_status()
            print(f"Successfully triggered dev_fleet for issue {report.issue_id}")
    except Exception as e:
        print(f"Failed to trigger dev_fleet webhook: {e}")

    return {"status": "success", "message": "Crash report received and forwarded"}

image = modal.Image.debian_slim().apt_install("git").pip_install(
    "fastapi[standard]",
    "pydantic",
    "deepdiff",
    "GitPython",
    "httpx"
)

@app.function(image=image, volumes={"/data": volume})
@modal.asgi_app()
def fastapi_app():
    # Ensure repos dir exists
    os.makedirs(REPOS_DIR, exist_ok=True)
    return web_app
