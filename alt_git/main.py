import modal
from fastapi import FastAPI
from pydantic import BaseModel

app = modal.App("alt-git-ledger")
web_app = FastAPI()

class CrashReport(BaseModel):
    issue_id: str
    traceback: str
    iteration: int

@web_app.post("/telemetry/crash")
async def receive_crash_report(report: CrashReport):
    print(f"Received crash report for issue {report.issue_id}, iteration {report.iteration}")
    print(f"Traceback: {report.traceback}")
    return {"status": "success", "message": "Crash report received"}

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "pydantic", "deepdiff")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
