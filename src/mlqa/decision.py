from dataclasses import dataclass
from src.mlqa.mlqa_client import analyze_patch


@dataclass
class HouseDecision:
    house_present: bool
    full_house: bool | None
    error: str | None


def decide(clean_path):
    qa = analyze_patch(clean_path)

    return HouseDecision(
        house_present=qa.get("house_present", False),
        full_house=qa.get("full_house_present"),
        error=qa.get("error_description"),
    )
