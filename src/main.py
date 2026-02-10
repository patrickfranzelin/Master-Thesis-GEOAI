from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
import os

from src.patches.extractor import extract_patch
from src.utils.geometry import polygon_to_sam_bbox
from src.utils.rendering import draw_points
from src.db.writer import write_mlqa
from src.patches.create_patch_output import create_patch_outputs

# New architecture imports
from src.pipeline.decision import mlqa_decide
from src.pipeline.routing import route_pipeline
from src.pipeline.full_house_pipeline import full_house_pipeline
from src.pipeline.partial_house_pipeline import partial_house_pipeline
from src.pipeline.discovery_pipeline import discovery_pipeline


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
    # Add bbox to debug image
    # ---------------------------------------------
    dbg = cv2.imread(str(debug_path))
    bbox = polygon_to_sam_bbox(poly_px)
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(str(debug_path), dbg)

    # ---------------------------------------------
    # DECISION STAGE: MLLM decides
    # ---------------------------------------------
    decision = mlqa_decide(clean_path)
    
    # ---------------------------------------------
    # ROUTING: Determine which pipeline to execute
    # ---------------------------------------------
    pipeline = route_pipeline(decision)
    
    print(f"  → Decision: house_present={decision.house_present}, full_house={decision.full_house}")
    print(f"  → Routing to: {pipeline} pipeline")
    
    # ---------------------------------------------
    # PIPELINE EXECUTION: Branch into three pipelines
    # ---------------------------------------------
    
    # Prepare paths for pipelines
    paths = {
        'clean': clean_path,
        'debug': debug_path,
        'raw': raw_path,
        'sam': sam_dir,
    }
    
    if pipeline == "FULL":
        # 🟢 FULL HOUSE PIPELINE
        qa, inside_pts, outside_pts = full_house_pipeline(img, poly_px, paths, row.id)
        
        # Optional debug: draw points
        if inside_pts or outside_pts:
            overlay = draw_points(cv2.imread(str(debug_path)), inside_pts, outside_pts)
            cv2.imwrite(str(points_dir / f"bld_{row.id:07d}_points.png"), overlay)
        
        # Prepare database record
        record = {
            "building_id": int(row.id),
            "patch_path": str(raw_path),
            "house_present": decision.house_present,
            "full_house_present": decision.full_house,
            "error_description": decision.reason,
            "inside_pts": inside_pts,
            "outside_pts": outside_pts,
        }
    
    elif pipeline == "PARTIAL":
        # 🟡 PARTIAL HOUSE PIPELINE
        qa, inside_pts, outside_pts, img_big, poly_big = partial_house_pipeline(
            row, gdf, paths, row.id
        )
        
        # Optional debug: draw points
        if inside_pts or outside_pts:
            overlay = draw_points(cv2.imread(str(debug_path)), inside_pts, outside_pts)
            cv2.imwrite(str(points_dir / f"bld_{row.id:07d}_points.png"), overlay)
        
        # Prepare database record
        record = {
            "building_id": int(row.id),
            "patch_path": str(raw_path),
            "house_present": decision.house_present,
            "full_house_present": decision.full_house,
            "error_description": decision.reason,
            "inside_pts": inside_pts,
            "outside_pts": outside_pts,
        }
    
    elif pipeline == "DISCOVERY":
        # 🔵 DISCOVERY PIPELINE
        buildings_found, negative_pts, discovered_polygons = discovery_pipeline(
            img, paths, row.id
        )
        
        # Prepare database record
        if len(discovered_polygons) > 0:
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
            record = {
                "building_id": int(row.id),
                "patch_path": str(raw_path),
                "house_present": False,
                "full_house_present": None,
                "error_description": "No buildings found in patch",
                "inside_pts": [],
                "outside_pts": [],
            }

    # ---------------------------------------------
    # Write DB (all pipelines)
    # ---------------------------------------------
    write_mlqa(record)

print("\nDONE")
