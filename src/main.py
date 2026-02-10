from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
import os

from src.db.loader import load_buildings
from src.patches.extractor import extract_patch
from src.utils.geometry import polygon_to_sam_bbox
from src.utils.rendering import (
    add_center_star,
    add_grid_overlay,
    add_polygon_overlay,
    draw_points,
)
from src.mlqa.mlqa_client import analyze_patch
from src.mlqa.point_client import analyze_points
from src.mlqa.discovery_client import discover_all_houses
from src.db.writer import write_mlqa
from src.patches.create_patch_output import create_patch_outputs
from src.mlqa.mlqa_stage import run_qa
from src.sam.sam_stage import run_sam_stage, run_sam_discovery


# --------------------------------------------------
# Paths
# --------------------------------------------------

output_dir = Path("../outputs/db_results")
sam_dir = output_dir / "sam"
sam_dir.mkdir(exist_ok=True)
raw_dir = output_dir / "raw"
clean_dir = output_dir / "clean"
debug_dir = output_dir / "debug"
points_dir = output_dir / "points"
out_dirs = {
    "raw": raw_dir,
    "clean": clean_dir,
    "debug": debug_dir,
}

for d in [raw_dir, clean_dir, debug_dir, points_dir]:
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load AOI from PostGIS (NOT ArcGIS)
# --------------------------------------------------

engine = create_engine(os.environ["PG_CONN"])

AOI_ID = 1

aoi_gdf = gpd.read_postgis(
    f"SELECT geom FROM src.aoi WHERE aoi_id = {AOI_ID}",
    engine,
    geom_col="geom",
)

if len(aoi_gdf) == 0:
    raise RuntimeError(f"AOI {AOI_ID} not found")

aoi_geom = aoi_gdf.geometry.iloc[0]



# --------------------------------------------------
# Load buildings intersecting AOI (SERVER SIDE)
# --------------------------------------------------

gdf = gpd.read_postgis(
    f"""
    SELECT id, geom, tiff_path
    FROM src.buildings
    WHERE tiff_path IS NOT NULL
      AND ST_Intersects(
            geom,
            (SELECT geom FROM src.aoi WHERE aoi_id = {AOI_ID})
          )
    """,
    engine,
    geom_col="geom",
)


print(f"Buildings inside AOI: {len(gdf)}")

if len(gdf) == 0:
    raise RuntimeError("AOI contains zero buildings.")

# --------------------------------------------------
# Main loop
# --------------------------------------------------

for _, row in gdf.iterrows():

    print(f"\nProcessing building {row.id}")

    # ---------------------------------------------
    # Extract patch
    # ---------------------------------------------
    img, poly_px = extract_patch(row.geom, gdf.crs, row.tiff_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ---------------------------------------------
    # Patch outputs
    # ---------------------------------------------
    raw_path, clean_path, debug_path = create_patch_outputs(
        img,
        poly_px,
        out_dirs,
        row.id,
    )
    print(f"Saved raw + clean + debug for {row.id}")

    # ---------------------------------------------
    # MLQA + point QA
    # ---------------------------------------------
    dbg = cv2.imread(str(debug_path))

    bbox = polygon_to_sam_bbox(poly_px)

    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(str(debug_path), dbg)

    qa, inside_pts, outside_pts = run_qa(clean_path, debug_path)

    # ---------------------------------------------
    # Optional debug: draw points
    # ---------------------------------------------
    if inside_pts or outside_pts:
        overlay = draw_points(cv2.imread(str(debug_path)), inside_pts, outside_pts)
        cv2.imwrite(str(points_dir / f"bld_{row.id:07d}_points.png"), overlay)
        print(f"Saved points overlay: bld_{row.id:07d}_points.png")

    # ---------------------------------------------
    # SAM refinement (with workflow separation)
    # ---------------------------------------------

    if qa["house_present"]:

        full_house = qa.get("full_house_present", True)
        
        if full_house:
            print(f"Building {row.id}: Full house detected - standard SAM workflow")
            # Use standard patch for full houses
            sam_img = img
            sam_poly = poly_px
            sam_mode = "standard"
            
        else:
            print(f"Building {row.id}: Partial house detected - escalated SAM workflow")
            # Extract larger patch for partial houses
            sam_img, sam_poly = extract_patch(
                row.geom,
                gdf.crs,
                row.tiff_path,
                context=5  # BIGGER PATCH for partial houses
            )
            sam_img = cv2.cvtColor(sam_img, cv2.COLOR_RGB2BGR)
            sam_mode = "escalated"

        run_sam_stage(
            sam_img,
            raw_path,
            sam_poly,
            inside_pts,
            outside_pts,
            sam_dir,
            row.id,
            mode=sam_mode
        )

    # ---------------------------------------------
    # Prepare database record based on workflow
    # ---------------------------------------------
    
    if not qa["house_present"]:
        # DISCOVERY MODE: No house in original polygon
        print(f"Building {row.id}: No house in polygon - running DISCOVERY mode")
        
        # Use MLQA to discover all buildings in the patch
        discovery_result = discover_all_houses(clean_path)
        
        buildings_found = discovery_result.get("buildings_found", [])
        negative_pts = discovery_result.get("negative_points", [])
        total = discovery_result.get("total_buildings", 0)
        
        print(f"  Discovery MLQA found {total} buildings in patch")
        
        if total > 0:
            # Run SAM in discovery mode to segment all found buildings
            discovered_polygons = run_sam_discovery(
                img,
                raw_path,
                buildings_found,
                negative_pts,
                sam_dir,
                row.id
            )
            
            # Store discovery results
            record = {
                "building_id": int(row.id),
                "patch_path": str(raw_path),
                "house_present": False,
                "full_house_present": None,
                "error_description": f"Discovery mode: found {len(discovered_polygons)} buildings",
                "inside_pts": [],
                "outside_pts": negative_pts,
            }
        else:
            print(f"  No buildings found in patch")
            record = {
                "building_id": int(row.id),
                "patch_path": str(raw_path),
                "house_present": False,
                "full_house_present": None,
                "error_description": "No buildings found in patch",
                "inside_pts": [],
                "outside_pts": [],
            }
    else:
        # STANDARD/ESCALATED MODE: House present in polygon
        record = {
            "building_id": int(row.id),
            "patch_path": str(raw_path),
            "house_present": qa["house_present"],
            "full_house_present": qa.get("full_house_present"),
            "error_description": qa["error_description"],
            "inside_pts": inside_pts,
            "outside_pts": outside_pts,
        }

    # ---------------------------------------------
    # Write DB (all workflows)
    # ---------------------------------------------
    write_mlqa(record)

print("\nDONE")
