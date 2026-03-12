# Modal Best Practices & Ground Truth

## Architecture & Code Organization
- **Avoid Circular Imports in Modal Apps:** When creating a Modal App (`app = modal.App("my_app")`), define it in a standalone module (e.g., `fleet_app.py`) rather than your main entry point `app.py`. Import this `app` instance across your codebase. This prevents `ModuleNotFoundError` circular imports during Modal's build and GPU memory snapshotting processes.
- **Isolate Web UI from GPU Inference:** Use lightweight CPU images (e.g., `modal.Image.debian_slim`) for web interfaces (like FastAPI). Call GPU-intensive orchestrators or inference functions via Modal-native RPC (`function.remote()`).
- **Volumes and Async:** When working with Modal Volumes inside FastAPI or other asynchronous endpoints, use asynchronous methods such as `await volume.commit.aio()` rather than blocking methods (`volume.commit()`) to prevent `AsyncUsageWarning` or event loop blocking.

## Modal 1.0 Deprecations & Modern Syntax
*As of Modal 1.0, several older patterns are deprecated and should be replaced with modern Image definitions.*

### Deprecated: `mount` and `modal.Mount`
In older versions, local files or Python packages were mounted to functions using `modal.Mount` and the `mount=` argument. This is now deprecated.

**❌ Bad (Deprecated):**
```python
import modal

mount = modal.Mount.from_local_dir("data")
mount = mount.add_local_file("config.yaml")

@app.function(image=image, mount=mount)
def f(): ...
```

**✅ Good (Modern):**
Attach files directly to the `Image` object.
```python
import modal

image = modal.Image.debian_slim().add_local_dir("data", remote_path="/root/data").add_local_file("config.yaml", remote_path="/root/config.yaml")

@app.function(image=image)
def f(): ...
```

### Deprecated: `Mount.from_local_python_packages`
To mount local python source code, do not use `modal.Mount`. Instead, use `Image.add_local_python_source`.

**❌ Bad (Deprecated):**
```python
import modal

mount = modal.Mount.from_local_python_packages("my_lib")

@app.function(image=image, mount=mount)
def f(): ...
```

**✅ Good (Modern):**
```python
import modal

image = modal.Image.debian_slim().add_local_python_source("my_lib")

@app.function(image=image)
def f(): ...
```

### Note on Image Copies
Avoid using `Image.copy_mount` as it has also been deprecated in favor of `add_local_file` and `add_local_dir`. If subsequent build steps require copied files, you can pass `copy=True` into `add_local_file` or `add_local_dir`.

## Deployment
- Deployment is typically executed via the CLI using `modal deploy my_app_entrypoint.py`.
- Ensure appropriate GitHub secrets or environment variables (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) are set prior to running deploy commands.

### Async Modal Methods
When invoking Modal functions from an asynchronous context (like an `async def` FastAPI endpoint), **never** use the synchronous `.remote()` method, as this will block the async event loop and raise an `InvalidError`. Instead, use the asynchronous `.remote.aio()` method.

**❌ Bad (Synchronous call in Async context):**
```python
@app.post("/")
async def run_endpoint(prompt: str):
    # This will crash with: InvalidError: You are running a Modal method synchronously inside an asynchronous context.
    result = run_agent.remote(prompt)
```

**✅ Good (Async call):**
```python
@app.post("/")
async def run_endpoint(prompt: str):
    # Use .aio() for async remote invocation
    result = await run_agent.remote.aio(prompt)
```
