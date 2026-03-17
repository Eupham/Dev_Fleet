import modal
from typing import List, Dict, Any, Type
from orchestrator.difficulty import estimate_difficulty # Uses your existing scoring logic

def query_llm(messages: List[Dict[str, str]], tier: str = None, schema: Any = None) -> str:
    """
    Dynamically routes the prompt to the correct GPU tier based on difficulty.
    """
    # 1. If no tier is forced, automatically calculate the best one
    if not tier:
        prompt_text = " ".join([m["content"] for m in messages])
        score = estimate_difficulty(prompt_text) # Returns 0.0 to 1.0
        
        if score < 0.3: tier = "trivial"   # T4 GPU (Cheap)
        elif score < 0.6: tier = "simple"  # L4 GPU
        else: tier = "moderate"           # L40S GPU (Powerful)

    # 2. Map the tier string to the correct Modal class
    class_map = {
        "trivial": ("InferenceSmall", "trivial"),
        "simple": ("InferenceMedium", "simple"),
        "moderate": ("Inference", "moderate"),
        "expert": ("InferenceLarge", "expert")
    }
    
    class_name, config_key = class_map.get(tier, ("Inference", "moderate"))
    
    # 3. Call the Modal class dynamically
    try:
        ModelClass = modal.Cls.from_name("dev-fleet", class_name)
        model_instance = ModelClass()
        return model_instance.generate.remote(messages=messages, schema=schema)
    except Exception as e:
        print(f"Routing failed to {tier}, falling back to Moderate: {e}")
        Fallback = modal.Cls.from_name("dev-fleet", "Inference")
        return Fallback().generate.remote(messages=messages, schema=schema)
