from pathlib import Path
from src.sam.model_sam3 import segment_with_text

TREE_PROMPT = ["tree"]

def segment_trees(image_path: Path):
    """
    Returns tree masks and polygons using SAM3 text prompting.
    """
    masks, polys = segment_with_text(
        image_path=image_path,
        text_prompts=TREE_PROMPT
    )

    if masks is None:
        return [], []

    return masks, polys