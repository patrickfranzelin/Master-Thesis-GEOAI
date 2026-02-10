"""
Full House Pipeline: Refines an already good footprint.

Goal: Use MLQA points to iteratively refine SAM segmentation of a complete house.

Characteristics:
- Normal patch size
- MLQA-generated points (inside/outside)
- Tight bbox around footprint
- Iterative SAM refinement (standard mode)
"""
from pathlib import Path
import cv2
import numpy as np

from src.mlqa.mlqa_stage import run_qa
from src.sam.sam_stage import run_sam_stage


def full_house_pipeline(img, poly_px, paths, bid):
    """
    Execute full house pipeline for complete buildings.
    
    Args:
        img: Input image (BGR format)
        poly_px: Polygon in pixel coordinates
        paths: Dict with 'clean', 'debug', 'sam', 'raw' paths
        bid: Building ID
        
    Returns:
        Tuple of (qa_result, inside_points, outside_points)
    """
    print(f"Building {bid}: Full house detected - standard SAM workflow")
    
    # Run MLQA to get house quality assessment and point placement
    qa, inside_pts, outside_pts = run_qa(paths['clean'], paths['debug'])
    
    # Run SAM in standard mode with MLQA points
    run_sam_stage(
        img=img,
        raw_path=paths['raw'],
        poly_px=poly_px,
        inside=inside_pts,
        outside=outside_pts,
        out_dir=paths['sam'],
        bid=bid,
        mode="standard"
    )
    
    return qa, inside_pts, outside_pts
