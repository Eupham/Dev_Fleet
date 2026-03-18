"""Bridge for Modal-hosted inference and heartbeat management."""
from __future__ import annotations
import modal
import threading
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

def ping_tier(tier: str = "moderate"):
    """Fires a background ping to prevent the 2-second scaledown window from closing."""
    class_name = TIER_MAP.get(tier, "Inference")
    try:
        ModelClass = modal.Cls.from_name("dev_fleet", class_name)
        # Call asynchronously so the orchestrator isn't blocked
        ModelClass().ping.spawn() 
    except Exception as e:
        print(f"⚠️ [Heartbeat] Failed to ping {class_name}: {e}")

def query_llm(messages: List[Dict[str, str]], tier: str = "moderate", schema: Any = None) -> Any:
    class_name = TIER_MAP.get(tier)
    if not class_name:
        raise ValueError(f"[ROUTING ERROR] Invalid tier '{tier}'")
        
    try:
        ModelClass = modal.Cls.from_name("dev_fleet", class_name)
        kwargs = {"messages": messages}
        if schema:
            kwargs["schema"] = schema
            
        return ModelClass().generate.remote(**kwargs)
    except Exception as e:
        raise RuntimeError(f"[INFERENCE ERROR] Failed on tier '{tier}': {e}")

def chat_completion(messages: List[Dict[str, str]], model: str = "llm", tier: str = "moderate", schema: Any = None) -> Any:
    return query_llm(messages, tier=tier, schema=schema)

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
