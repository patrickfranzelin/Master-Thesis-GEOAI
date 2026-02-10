"""
Decision module: MLLM makes the core decision about house presence and completeness.
"""
from dataclasses import dataclass
from pathlib import Path
from src.mlqa.mlqa_client import analyze_patch


@dataclass
class HouseDecision:
    """
    Single source of truth for MLLM decision.
    
    Attributes:
        house_present: Whether any house exists in the polygon
        full_house: Whether the polygon covers nearly all of the house (None if no house)
        reason: Error description or None if no error
    """
    house_present: bool
    full_house: bool | None
    reason: str | None


def mlqa_decide(clean_patch: Path) -> HouseDecision:
    """
    Run MLLM analysis on the clean patch and return structured decision.
    
    Args:
        clean_patch: Path to the clean patch image
        
    Returns:
        HouseDecision object with MLLM analysis results
    """
    qa = analyze_patch(clean_patch)
    
    return HouseDecision(
        house_present=qa["house_present"],
        full_house=qa.get("full_house_present"),
        reason=qa.get("error_description"),
    )
