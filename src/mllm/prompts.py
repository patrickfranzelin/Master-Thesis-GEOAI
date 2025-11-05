def points_prompt() -> str:
    return (
        "<image>\n"
        "You are a precise visual inspector.\n"
        "Look at the red polygon in this aerial image.\n"
        "Output exactly eight 2D pixel coordinates in JSON format like this:\n"
        "{'inside': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 'outside': [[x5, y5], [x6, y6], [x7, y7], [x8, y8]]}\n"
        "Rules:\n"
        "- Each coordinate must be two integers.\n"
        "- Choose 4 points well distributed inside the polygon (on the roof, not clustered).\n"
        "- Choose 4 points clearly outside the polygon (on ground/road, not clustered).\n"
        "- Output only the JSON object. No extra text."
    )
