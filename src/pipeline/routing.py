"""
Routing module: Routes to appropriate pipeline based on MLLM decision.
"""
from src.pipeline.decision import HouseDecision


def route_pipeline(decision: HouseDecision) -> str:
    """
    Determine which pipeline to execute based on MLLM decision.
    
    Args:
        decision: HouseDecision object from MLLM
        
    Returns:
        Pipeline name: "FULL", "PARTIAL", or "DISCOVERY"
    """
    if not decision.house_present:
        return "DISCOVERY"
    
    if decision.full_house:
        return "FULL"
    
    return "PARTIAL"
