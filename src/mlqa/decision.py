from dataclasses import dataclass
from src.mlqa.mlqa_client import analyze_patch
from src.mlqa.error_client import analyze_start_polygon


@dataclass
class HouseDecision:
    house_present: bool
    full_house: bool | None
    error: str | None
    errors: list | None = None


def decide(clean_path):

    # --------------------------
    # 1. MLQA (presence + coverage)
    # --------------------------
    qa = analyze_patch(clean_path)

    house_present = qa.get("house_present", False)
    full_house = qa.get("full_house_present")

    # --------------------------
    # 2. If no house → skip error analysis
    # --------------------------
    if not house_present:
        return HouseDecision(
            house_present=False,
            full_house=False,
            error="No building detected",
            errors=[]
        )

    # --------------------------
    # 3. Error analysis (NEW SYSTEM)
    # --------------------------
    error_eval = analyze_start_polygon(clean_path)

    tags = error_eval.get("tags", [])

    # add MISALIGNED explicitly
    if error_eval.get("misaligned"):
        tags = ["MISALIGNED"] + tags

    description = error_eval.get("description", "")

    # --------------------------
    # 4. Final decision
    # --------------------------
    return HouseDecision(
        house_present=True,
        full_house=full_house,
        error=description,
        errors=tags,
    )