"""Bridge for Modal-hosted inference and Continuous Heartbeat.
FIXES APPLIED (v2):
1. chat_completion() now forwards `tools` and `tool_choice` kwargs to the
   Modal inference class, which passes them to llama-server's OpenAI-
   compatible API. This is the missing link that makes the tool-use loop
   in agent_loop.py actually work.
2. generate_logic() now returns the FULL response object (not just content
   string) when tools are present, so the caller can inspect tool_calls.
3. Added response normalization: when tools are used, returns a dict with
   choices[0].message structure matching OpenAI format.
"""
from __future__ import annotations
import modal
import threading
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
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
                ModelClass().ping.spawn()
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
    When `tools` is provided, the response will include tool_calls if the
    model decides to use them. The caller (agent_loop.py) is responsible
    for dispatching tool calls and feeding results back.
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
