from src.pipelines.full_house import FullHousePipeline
from src.pipelines.partial_house import PartialHousePipeline

def route(decision):
    """
    Route to appropriate pipeline based on MLQA decision.
    
    Explicit handling of all cases:
    - house_present=False → None (no pipeline)
    - full_house_present=True → FullHousePipeline
    - full_house_present=False → PartialHousePipeline  
    - full_house_present=None → PartialHousePipeline (uncertain → use discovery)
    """
    if not decision.house_present:
        return None
    
    # Explicit routing based on full_house status
    if decision.full_house is True:
        return FullHousePipeline()
    elif decision.full_house is False:
        return PartialHousePipeline()
    elif decision.full_house is None:
        # Uncertain case: use discovery mode to be safe
        return PartialHousePipeline()
    
    # Fallback: if somehow we get here, use partial
    return PartialHousePipeline()
