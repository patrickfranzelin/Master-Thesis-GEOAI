from src.pipelines.full_house import FullHousePipeline
from src.pipelines.partial_house import PartialHousePipeline


def route(decision):
    """
    Routing logic:

    - house_present=False → None
    - full_house=True → FullHousePipeline
    - full_house=False → PartialHousePipeline
    - full_house=None → PartialHousePipeline
    """

    if not decision.house_present:
        return None

    if decision.full_house is True:
        return FullHousePipeline()

    # partial OR uncertain → use partial pipeline
    return PartialHousePipeline()
