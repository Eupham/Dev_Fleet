import modal
from typing import List, Dict, Any

def query_llm(messages: List[Dict[str, str]], tier: str = "moderate", schema: Any = None) -> str:
    """
    Connects to the deployed Modal Inference class and generates a response.
    Strictly enforces the requested tier. No silent fallbacks.
    """
    # Updated to perfectly match the 5 outputs of difficulty_to_tier()
    class_map = {
        "trivial": "InferenceSmall",   # T4 GPU
        "simple": "InferenceMedium",   # L4 GPU
        "moderate": "Inference",       # L40S GPU
        "complex": "Inference",        # L40S GPU (Bridging the gap)
        "expert": "InferenceLarge"     # L40S GPU
    }
    
    # 1. Fail loud if an unknown tier is passed
    if tier not in class_map:
        raise ValueError(
            f"[ROUTING ERROR] Invalid tier '{tier}' requested. "
            f"Must be one of: {list(class_map.keys())}"
        )
        
    class_name = class_map[tier]
    
    # 2. Fail loud if Modal cannot connect to the specific class
    try:
        ModelClass = modal.Cls.from_name("dev_fleet", class_name)
        model_instance = ModelClass()
    except modal.exception.NotFoundError:
        raise RuntimeError(
            f"[MODAL ERROR] Could not find class '{class_name}' in app 'dev_fleet'. "
            f"Make sure your app is currently deployed and the class name matches."
        )
    except Exception as e:
        raise RuntimeError(
            f"[INITIALIZATION ERROR] Failed to connect to tier '{tier}' ({class_name}). "
            f"Details: {e}"
        )

    # 3. Fail loud if the generation itself crashes
    try:
        return model_instance.generate.remote(messages=messages, schema=schema)
    except Exception as e:
        raise RuntimeError(
            f"[INFERENCE ERROR] Generation failed on tier '{tier}' ({class_name}). "
            f"Details: {e}"
        )
