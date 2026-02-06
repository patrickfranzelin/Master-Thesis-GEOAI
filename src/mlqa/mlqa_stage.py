from src.mlqa.mlqa_client import analyze_patch
from src.mlqa.point_client import analyze_points


def run_qa(clean_path, debug_path):

    qa = analyze_patch(clean_path)

    inside = []
    outside = []

    if qa["house_present"]:
        pts = analyze_points(debug_path)
        inside = pts.get("inside", [])
        outside = pts.get("outside", [])

    return qa, inside, outside
