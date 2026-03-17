import modal
from typing import List, Dict, Any

def query_llm(messages: List[Dict[str, str]], tier: str = "moderate", schema: Any = None) -> Any:
    """Connects to the deployed Modal Inference class and generates a response."""
    
    # Map all 5 tiers from difficulty.py to the 4 Modal classes
    class_map = {
        "trivial": "InferenceSmall",
        "simple": "InferenceMedium",
        "moderate": "Inference",
        "complex": "Inference",       # Safely mapped to your L40S
        "expert": "InferenceLarge"
    }
    
    class_name = class_map.get(tier)
    if not class_name:
        raise ValueError(
            f"[ROUTING ERROR] Invalid tier '{tier}' requested. "
            f"Must be one of: {list(class_map.keys())}"
        )
        
    try:
        # Look up the deployed class on the dev_fleet app
        ModelClass = modal.Cls.from_name("dev_fleet", class_name)
        
        # Instantiate and call the remote method securely
        kwargs = {"messages": messages}
        if schema:
            kwargs["schema"] = schema
            
        return ModelClass().generate.remote(**kwargs)
        
    except modal.exception.NotFoundError:
        raise RuntimeError(
            f"[MODAL ERROR] Could not find class '{class_name}' in app 'dev_fleet'. "
            f"Ensure `modal deploy app.py` has finished successfully."
        )
    except Exception as e:
        raise RuntimeError(f"[INFERENCE ERROR] Failed on tier '{tier}' ({class_name}): {e}")


def chat_completion(messages: List[Dict[str, str]], model: str = "llm", temperature: float = 0.0, max_tokens: int = 2048, schema: Any = None) -> Any:
    """
    Compatibility alias for orchestrator/task_parser.py 
    Routes parsing tasks to the moderate tier by default.
    """
    return query_llm(messages, tier="moderate", schema=schema)
