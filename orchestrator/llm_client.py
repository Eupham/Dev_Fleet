"""LLM Client — Thin wrapper around Modal-native inference calls.

All orchestrator modules call ``chat_completion()`` which routes to
``Inference().generate.remote()`` — no HTTP overhead, no idle timeouts.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


def chat_completion(
    messages: list[dict[str, str]],
    model: str = "llm",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    schema: Optional[type[BaseModel]] = None,
) -> Any:
    """Send a chat completion request via Modal-native RPC.

    Returns the generated text from the model, or a Pydantic object if schema is provided.
    """
    import modal
    import asyncio

    # Try importing the Inference class directly from the local app for ephemeral runs.
    # Fallback to dynamically loading the remote class from the deployed "dev_fleet" app
    # when called from isolated container contexts (like the test runner or chainlit).
    try:
        from inference.server import Inference
        inference_inst = Inference()
    except Exception:
        inference_inst = modal.Cls.from_name("dev_fleet", "Inference")()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside a running event loop (e.g. LlamaIndex async pipeline).
        # We cannot use loop.run_until_complete() or asyncio.run() here.
        # We must use the blocking remote call because this function is fundamentally synchronous
        # in the context of `def complete(self, ...)` in CustomLLM.
        # This may emit an AsyncUsageWarning from Modal, but it will execute.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return inference_inst.generate.remote(
                messages, model=model, temperature=temperature, max_tokens=max_tokens, schema=schema
            )
    else:
        # No running event loop. Safe to use blocking .remote()
        return inference_inst.generate.remote(
            messages, model=model, temperature=temperature, max_tokens=max_tokens, schema=schema
        )


def generate(
    context: str,
    task_description: str,
    model: str = "llm",
    temperature: float = 0.3,
) -> str:
    """Query the vLLM inference service for a code/plan response.

    Parameters
    ----------
    context:
        GraphRAG context window built from the Tri-Graph memory.
    task_description:
        The atomic task description.
    model:
        Served model alias.
    temperature:
        Sampling temperature (lower = more deterministic).

    Returns
    -------
    The generated text from the model.
    """
    system = (
        "You are a senior software engineer. Use the CONTEXT below to "
        "complete the TASK. Respond with code or a precise plan.\n\n"
        f"CONTEXT:\n{context}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_description},
    ]
    return chat_completion(messages, model=model, temperature=temperature, max_tokens=4096)


class ModalVLLM(CustomLLM):
    """Custom LlamaIndex LLM wrapper that routes to our Modal vLLM engine."""

    context_window: int = 8192
    num_output: int = 4096
    model_name: str = "llm"
    temperature: float = 0.3

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response_text = chat_completion(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.num_output,
        )
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Streaming is not supported via this simple RPC wrapper.")
