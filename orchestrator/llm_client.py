"""Bridge for Modal-hosted inference and Continuous Heartbeat.

FIXES APPLIED (v3):
1. chat_completion() and query_llm() forward `tools` and `tool_choice` to Modal.
2. Modal RPC error handling: catches deserialization failures (the openai
   APIStatusError can't be unpickled across Modal boundary due to missing
   'response'/'body' kwargs). Re-raises as clean RuntimeError.
3. ModalKeepAlive: continuous heartbeat to hold scaledown window open.
"""
from __future__ import annotations
import modal
import threading
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

logger = logging.getLogger("dev_fleet.orchestrator")

TIER_MAP = {
    "trivial": "InferenceSmall",
    "simple": "InferenceMedium",
    "moderate": "Inference",
    "complex": "Inference",
    "expert": "InferenceLarge"
}

class ModalKeepAlive:
    """
    INFRASTRUCTURE FIX: A background thread that continuously pings Modal
    to hold the 2-second scaledown window open during local computation.
    """
    def __init__(self, tier: str = "moderate"):
        self.tier = tier
        self.running = False
        self.thread = None

    def _ping_loop(self):
        class_name = TIER_MAP.get(self.tier, "Inference")
        try:
            ModelClass = modal.Cls.from_name("dev_fleet", class_name)
            while self.running:
                try:
                    ModelClass().ping.spawn()
                except Exception:
                    pass  # Individual pings can fail without stopping the loop
                time.sleep(1.0)  # Pulse faster than the 2s timeout
        except Exception:
            pass

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

def query_llm(
    messages: List[Dict[str, str]],
    tier: str = "moderate",
    schema: Any = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Any] = None,
) -> Any:
    class_name = TIER_MAP.get(tier)
    if not class_name:
        raise ValueError(f"[ROUTING ERROR] Invalid tier '{tier}'")

    try:
        ModelClass = modal.Cls.from_name("dev_fleet", class_name)
        kwargs = {"messages": messages}
        if schema:
            kwargs["schema"] = schema
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        return ModelClass().generate.remote(**kwargs)

    except modal.exception.ExecutionError as e:
        # CRITICAL FIX: Modal can't deserialize openai.APIStatusError across
        # the RPC boundary because it requires 'response' and 'body' kwargs
        # that don't survive pickling. Extract the useful error message.
        error_msg = str(e)
        if "deserialize" in error_msg.lower():
            # The remote side already wrapped in RuntimeError (our utils.py fix),
            # but if an older version is deployed, handle it gracefully
            raise RuntimeError(
                f"[INFERENCE ERROR] Remote exception on tier '{tier}' "
                f"(deserialization failed — check that inference/utils.py wraps "
                f"exceptions in RuntimeError): {error_msg[:300]}"
            )
        raise RuntimeError(f"[INFERENCE ERROR] Failed on tier '{tier}': {error_msg[:300]}")

    except Exception as e:
        raise RuntimeError(f"[INFERENCE ERROR] Failed on tier '{tier}': {e}")

def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "llm",
    tier: str = "moderate",
    schema: Any = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Any] = None,
) -> Any:
    """Route a chat completion request to the appropriate Modal inference tier.

    When `tools` is provided, the response will be a dict with
    choices[0].message.tool_calls if the model decides to use them.
    The caller (agent_loop.py) is responsible for dispatching tool calls
    and feeding results back.
    """
    return query_llm(messages, tier=tier, schema=schema, tools=tools, tool_choice=tool_choice)

class DevFleetLLM(CustomLLM):
    context_window: int = 32768
    num_output: int = 2048
    model_name: str = "dev-fleet-llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window, num_output=self.num_output, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=str(query_llm([{"role": "user", "content": prompt}], tier="moderate")))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Streaming is not implemented.")
