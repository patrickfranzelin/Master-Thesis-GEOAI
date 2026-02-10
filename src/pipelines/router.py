from src.pipelines.full_house import FullHousePipeline
from src.pipelines.partial_house import PartialHousePipeline

def route(decision):
    if not decision.house_present:
        return None
    if decision.full_house:
        return FullHousePipeline()
    return PartialHousePipeline()
