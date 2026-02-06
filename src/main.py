from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
import os

from src.db.loader import load_buildings
from src.patches.extractor import extract_patch
from src.utils.rendering import (
    add_center_star,
    add_grid_overlay,
    add_polygon_overlay,
    draw_points,
)
from src.mlqa.mlqa_client import analyze_patch
from src.mlqa.point_client import analyze_points
from src.db.writer import write_mlqa
from src.sam.sam_client import run_sam
from src.utils.geometry import polygon_to_sam_bbox



# --------------------------------------------------
# Paths
# --------------------------------------------------

output_dir = Path("../outputs/db_results")

raw_dir = output_dir / "raw"
clean_dir = output_dir / "clean"
debug_dir = output_dir / "debug"
points_dir = output_dir / "points"

for d in [raw_dir, clean_dir, debug_dir, points_dir]:
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load AOI from PostGIS (NOT ArcGIS)
# --------------------------------------------------

engine = create_engine(os.environ["PG_CONN"])

AOI_ID = 3

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

    img, poly_px = extract_patch(row.geom, gdf.crs, row.tiff_path)

    # --------------------------------------------------
    # Convert ONCE to OpenCV BGR
    # --------------------------------------------------
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ==================================================
    # RAW PATCH
    # ==================================================

    raw_path = raw_dir / f"bld_{row.id:07d}_raw.png"
    cv2.imwrite(str(raw_path), img)

    # ==================================================
    # CLEAN PATCH
    # ==================================================

    clean_img = add_polygon_overlay(img.copy(), poly_px)
    clean_path = clean_dir / f"bld_{row.id:07d}_clean.png"
    cv2.imwrite(str(clean_path), clean_img)

    # ==================================================
    # DEBUG PATCH
    # ==================================================

    debug_img = add_polygon_overlay(img.copy(), poly_px)
    debug_img = add_center_star(debug_img)
    debug_img = add_grid_overlay(debug_img)

    debug_path = debug_dir / f"bld_{row.id:07d}_debug.png"
    cv2.imwrite(str(debug_path), debug_img)

    print(f"Saved raw + clean + debug for {row.id}")

    # ==================================================
    # MLQA — RAW ONLY
    # ==================================================

    qa = analyze_patch(clean_path)
    print("QA:", qa)

    inside_pts = []
    outside_pts = []

    # ==================================================
    # POINT QA
    # ==================================================

    if qa["house_present"]:

        pts = analyze_points(debug_path)
        inside_pts = pts.get("inside", [])
        outside_pts = pts.get("outside", [])

        overlay = draw_points(debug_img.copy(), inside_pts, outside_pts)

        points_path = points_dir / f"bld_{row.id:07d}_points.png"
        cv2.imwrite(str(points_path), overlay)

        print(f"Saved points overlay: {points_path.name}")

        # ==================================================
        # SAM REFINEMENT (after points) — POINTS ONLY
        # ==================================================

        if len(inside_pts) >= 1:

            sam_dir = output_dir / "sam"
            sam_dir.mkdir(exist_ok=True)

            # ---------------------------------------------
            # Debug visualization (points only)
            # ---------------------------------------------

            sam_input = img.copy()

            for x, y in inside_pts:
                cv2.circle(sam_input, (int(x), int(y)), 6, (0, 255, 0), -1)

            for x, y in outside_pts:
                cv2.circle(sam_input, (int(x), int(y)), 6, (0, 0, 255), -1)

            cv2.imwrite(str(sam_dir / f"bld_{row.id:07d}_sam_input.png"), sam_input)

            # ---------------------------------------------
            # RUN SAM (raw image + point prompts)
            # ---------------------------------------------

            mask, sam_poly = run_sam(
                raw_path,
                inside_pts,
                outside_pts,
            )

            if mask is not None:

                cv2.imwrite(str(sam_dir / f"bld_{row.id:07d}_mask.png"), mask)

                if sam_poly is not None:
                    overlay = img.copy()
                    pts = np.array(sam_poly.exterior.coords).astype("int32")
                    cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

                    cv2.imwrite(str(sam_dir / f"bld_{row.id:07d}_sam.png"), overlay)

                print("SAM refined:", row.id)

    # ==================================================
    # WRITE DATABASE
    # ==================================================

    record = {
        "building_id": int(row.id),
        "patch_path": str(raw_path),
        "house_present": qa["house_present"],
        "error_description": qa["error_description"],
        "inside_pts": inside_pts,
        "outside_pts": outside_pts,
    }

    write_mlqa(record)



print("\nDONE")
