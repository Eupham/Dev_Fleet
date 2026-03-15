"""LLM Client — Thin wrapper around Modal-native inference calls.

All orchestrator modules call ``chat_completion()`` which routes to the
appropriate tier's ``generate.remote()`` call — no HTTP overhead, no idle timeouts.

Tier routing:
  trivial  → InferenceSmall  (Qwen3-4B, T4)
  simple   → InferenceMedium (Qwen3-8B, T4)
  moderate → Inference        (Qwen3-Coder-30B-A3B, A10G)  [default]
  complex  → Inference        (Qwen3-Coder-30B-A3B, A10G)
  expert   → InferenceLarge   (Qwen3-Coder-480B-A35B, A100)
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


def _get_inference_instance(tier: str = "moderate"):
    """Return the appropriate inference instance for the given tier.

    Falls back to the primary Inference class on any error.
    """
    import modal
    import warnings

    # Moderate and complex both use the primary A10G Inference class
    if tier in ("moderate", "complex", None, ""):
        try:
            from inference.server import Inference
            return Inference()
        except Exception:
            return modal.Cls.from_name("dev_fleet", "Inference")()

    if tier == "trivial":
        try:
            from inference.model_pool import InferenceSmall
            return InferenceSmall()
        except Exception:
            try:
                return modal.Cls.from_name("dev_fleet", "InferenceSmall")()
            except Exception:
                pass

    if tier == "simple":
        try:
            from inference.model_pool import InferenceMedium
            return InferenceMedium()
        except Exception:
            try:
                return modal.Cls.from_name("dev_fleet", "InferenceMedium")()
            except Exception:
                pass

    if tier == "expert":
        try:
            from inference.model_pool import InferenceLarge
            return InferenceLarge()
        except Exception:
            try:
                return modal.Cls.from_name("dev_fleet", "InferenceLarge")()
            except Exception:
                pass

    # Fallback to primary Inference for any unrecognized tier
    try:
        from inference.server import Inference
        return Inference()
    except Exception:
        return modal.Cls.from_name("dev_fleet", "Inference")()


def chat_completion(
    messages: list[dict[str, str]],
    model: str = "llm",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    schema: Optional[type[BaseModel]] = None,
    tier: str = "moderate",
) -> Any:
    """Send a chat completion request via Modal-native RPC.

    Parameters
    ----------
    messages:
        OpenAI-format message list.
    model:
        Served model alias (default "llm").
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens to generate.
    schema:
        Optional Pydantic model for structured JSON output.
    tier:
        Routing tier: trivial | simple | moderate | complex | expert.
        Determines which GPU class handles the request.

    Returns
    -------
    Generated text, or Pydantic object if schema is provided.
    """
    import asyncio
    import warnings

    inference_inst = _get_inference_instance(tier)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return inference_inst.generate.remote(
                messages, model=model, temperature=temperature,
                max_tokens=max_tokens, schema=schema
            )
    else:
        return inference_inst.generate.remote(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, schema=schema
        )


def generate(
    context: str,
    task_description: str,
    model: str = "llm",
    temperature: float = 0.3,
    tier: str = "moderate",
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
    tier:
        Routing tier for model selection.

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
    return chat_completion(
        messages, model=model, temperature=temperature, max_tokens=4096, tier=tier
    )


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
            tier="moderate",
        )
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Streaming is not supported via this simple RPC wrapper.")
