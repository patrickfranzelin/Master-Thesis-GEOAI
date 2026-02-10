"""
Partial House Pipeline: Recovers complete house from incomplete polygon.

Goal: When polygon is bad (cuts off part of house), extract larger patch
      and use escalated SAM strategy to recover the whole building.

Key differences from full house:
- BIGGER patch (larger context)
- Escalated SAM mode (larger bbox, optional points)
- Focus on bbox-driven segmentation
"""
from pathlib import Path
import cv2

from src.patches.extractor import extract_patch
from src.mlqa.mlqa_stage import run_qa
from src.sam.sam_stage import run_sam_stage


def partial_house_pipeline(row, gdf, paths, bid):
    """
    Execute partial house pipeline for incomplete building footprints.
    
    Args:
        row: GeoDataFrame row with building geometry and metadata
        gdf: Full GeoDataFrame (for CRS)
        paths: Dict with 'clean', 'debug', 'sam', 'raw' paths
        bid: Building ID
        
    Returns:
        Tuple of (qa_result, inside_points, outside_points, larger_img, larger_poly)
    """
    print(f"Building {bid}: Partial house detected - escalated SAM workflow")
    
    # Run MLQA on original patch (still useful for analysis)
    qa, inside_pts, outside_pts = run_qa(paths['clean'], paths['debug'])
    
    # Extract LARGER patch for escalated processing
    img_big, poly_big = extract_patch(
        row.geom,
        gdf.crs,
        row.tiff_path,
        context=5  # BIGGER PATCH for partial houses
    )
    img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)
    
    # Run SAM in escalated mode (bbox-focused, resets MLQA points)
    run_sam_stage(
        img=img_big,
        raw_path=paths['raw'],
        poly_px=poly_big,
        inside=inside_pts,  # Will be reset in escalated mode
        outside=outside_pts,  # Will be reset in escalated mode
        out_dir=paths['sam'],
        bid=bid,
        mode="escalated"
    )
    
    return qa, inside_pts, outside_pts, img_big, poly_big
