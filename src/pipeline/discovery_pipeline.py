"""
Discovery Pipeline: Finds all houses when footprint is completely wrong.

Goal: When no house exists in the polygon, search the entire patch for 
      ANY buildings and segment them all.

This is a fundamentally different problem:
- No anchor polygon to work from
- Multi-building detection
- Exploratory search pattern
- Simplified MLQA prompt for 8b model
"""
from pathlib import Path
import cv2

from src.mlqa.discovery_client import discover_all_houses
from src.sam.sam_stage import run_sam_discovery


def discovery_pipeline(img, paths, bid):
    """
    Execute discovery pipeline to find all buildings in the patch.
    
    Args:
        img: Input image (BGR format)
        paths: Dict with 'clean', 'sam', 'raw' paths
        bid: Building ID
        
    Returns:
        Tuple of (buildings_found, negative_points, discovered_polygons)
    """
    print(f"Building {bid}: No house in polygon - running DISCOVERY mode")
    
    # Use MLQA to discover all buildings in the patch
    discovery_result = discover_all_houses(paths['clean'])
    
    buildings_found = discovery_result.get("buildings_found", [])
    negative_pts = discovery_result.get("negative_points", [])
    total = discovery_result.get("total_buildings", 0)
    
    print(f"  Discovery MLQA found {total} buildings in patch")
    
    discovered_polygons = []
    
    if total > 0:
        # Run SAM in discovery mode to segment all found buildings
        discovered_polygons = run_sam_discovery(
            img,
            paths['raw'],
            buildings_found,
            negative_pts,
            paths['sam'],
            bid
        )
        print(f"  SAM successfully segmented {len(discovered_polygons)} buildings")
    else:
        print(f"  No buildings found in patch")
    
    return buildings_found, negative_pts, discovered_polygons
