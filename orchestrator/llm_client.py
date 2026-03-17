import modal
from typing import List, Dict, Any

def query_llm(messages: List[Dict[str, str]], tier: str = "moderate") -> str:
    """Connects to the deployed Modal Inference class and generates a response."""
    try:
        # Map tiers to Modal class names
        class_map = {
            "trivial": "InferenceSmall",
            "simple": "InferenceMedium",
            "moderate": "Inference",
            "expert": "InferenceLarge"
        }
        
        # Look up the deployed class on the dev_fleet app
        Inference = modal.Cls.from_name("dev_fleet", class_map.get(tier, "Inference"))
        
        # Instantiate and call the remote method
        return Inference().generate.remote(messages=messages)
    except Exception as e:
        return f"Error connecting to inference engine: {str(e)}"
