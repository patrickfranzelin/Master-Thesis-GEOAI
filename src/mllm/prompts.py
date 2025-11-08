def points_prompt() -> str:
    return (
        "<image>\n"
        "You are a precise visual inspector.\n"
        "Look at the aerial image with a yellow polygon and numbered grid overlay.\n"
        "Use the grid coordinates as a reference frame.\n"
        "Output exactly eight 2D pixel coordinates in JSON format like this:\n"
        "{'inside': [[x1, y1], ..., [x4, y4]], 'outside': [[x5, y5], ..., [x8, y8]]}\n"
        "Rules:\n"
        "- Each coordinate must be two integers.\n"
        "- Choose 4 points well distributed inside the polygon (on roofs if visible).\n"
        "- Choose 4 points outside, near but clearly not on the roof.\n"
        "- Do not output anything if no buildings are visible."
    )
