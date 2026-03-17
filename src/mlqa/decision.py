from dataclasses import dataclass
from src.mlqa.mlqa_client import analyze_patch
from src.mlqa.error_client import analyze_errors

@dataclass
class HouseDecision:
    house_present: bool
    full_house: bool | None
    error: str | None
    errors: list | None = None


def decide(clean_path):
    qa = analyze_patch(clean_path)
    #error_detail = analyze_errors(clean_path)
    return HouseDecision(
        house_present=qa.get("house_present", False),
        full_house=qa.get("full_house_present"),
        error=qa.get("error_description"),
        #errors=error_detail.get("errors", []),

    )
