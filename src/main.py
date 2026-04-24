from pathlib import Path
import cv2
import geopandas as gpd
from sqlalchemy import create_engine
import os
from datetime import datetime
from uuid import uuid4
import torch

from src.core.context import PipelineContext
from src.db.export_to_filegdb import export_buildings_to_filegdb
from src.mlqa.decision import decide
from src.mlqa.mlqa_client import MLQAParseError
from src.pipelines.router import route
from src.patches.extractor import extract_patch, extract_aoi_raster
from src.patches.create_patch_output import create_patch_outputs
from src.db.writer import write_mlqa, write_detected_trees
from src.db.writer import write_detected_houses
from src.pipelines.global_discovery import run_global_discovery
from src.postprocess.matching import filter_new_buildings
from src.postprocess.deduplication import deduplicate_polygons
from src.utils.geometry import pixel_to_world
from src.mlqa.error_client import analyze_start_polygon
# --------------------------------------------------
# Paths
# --------------------------------------------------

output_dir = Path("../outputs/db_results")
sam_dir = output_dir / "sam"
raw_dir = output_dir / "raw"
clean_dir = output_dir / "clean"
debug_dir = output_dir / "debug"
comparison_dir = output_dir / "comparison"
ENABLE_GLOBAL = True

print(torch.cuda.is_available())

for d in [sam_dir, raw_dir, clean_dir, debug_dir, comparison_dir]:
    d.mkdir(parents=True, exist_ok=True)

out_dirs = {
    "raw": raw_dir,
    "clean": clean_dir,
    "debug": debug_dir,
    "comparison": comparison_dir,
}

RUN_ID = str(uuid4())
print(f"RUN_ID: {RUN_ID}")
# --------------------------------------------------
# Database
# --------------------------------------------------

engine = create_engine(os.environ["PG_CONN"])
AOI_ID = 3

aoi_gdf = gpd.read_postgis(
    f"SELECT geom FROM src.aoi WHERE aoi_id = {AOI_ID}",
    engine,
    geom_col="geom",
)

if aoi_gdf.empty:
    raise RuntimeError(f"AOI {AOI_ID} not found")

gdf = gpd.read_postgis(
    f"""
    SELECT id, geom, tiff_path
    FROM src.buildings
    WHERE tiff_path IS NOT NULL
      AND ST_Intersects(
            geom,
            (SELECT geom FROM src.aoi WHERE aoi_id = {AOI_ID})
          )
    LIMIT 1000
    """,
    engine,
    geom_col="geom",
)

if gdf.empty:
    raise RuntimeError("AOI contains zero buildings")

print(f"Buildings inside AOI: {len(gdf)}")

# --------------------------------------------------
# Main loop
# --------------------------------------------------
all_refined_polys = []
images = {}
for _, row in gdf.iterrows():

    print(f"\nProcessing building {row.id}")

    # ---------------------------------------------
    # Patch extraction
    # ---------------------------------------------

    img, poly_px, win = extract_patch(row.geom, gdf.crs, row.tiff_path, context=1.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    raw_path, clean_path, debug_path = create_patch_outputs(
        img,
        poly_px,
        out_dirs,
        row.id,
    )

    # ---------------------------------------------
    # MLQA decision
    # ---------------------------------------------

    try:
        decision = decide(clean_path)
    except MLQAParseError as e:
        # Parse error should abort, not create false negative
        print(f" MLQA parse error for building {row.id}: {e}")
        write_mlqa({
            "building_id": row.id,
            "patch_path": str(clean_path),
            "house_present": None,  # Indicate uncertainty
            "full_house_present": None,
            "error_description": f"MLQA_PARSE_ERROR: {str(e)}",
            "inside_pts": [],
            "outside_pts": [],
        })
        continue

    pipeline = route(decision)

    ctx = PipelineContext(
        building_id=row.id,
        img=img,
        poly_px=poly_px,
        raw_path=raw_path,
        clean_path=clean_path,
        debug_path=debug_path,
        sam_dir=sam_dir,
        geom=row.geom,
        crs=gdf.crs,
        tiff_path=row.tiff_path,
    )

    # ---------------------------------------------
    # No house → record error only
    # ---------------------------------------------

    if pipeline is None:
        write_mlqa({
            "building_id": row.id,
            "patch_path": str(clean_path),
            "house_present": False,
            "full_house_present": None,
            "error_description": decision.error,
            "inside_pts": [],
            "outside_pts": [],
        })
        continue

    # ---------------------------------------------
    # Execute pipeline and capture results (fixes Bug 3)
    # ---------------------------------------------
    result = pipeline.execute(ctx)

    # Normalize result.sam_polygons into list
    if result.sam_polygons:

        # Determine detection type
        if decision.full_house is True:
            dtype = "full"
        elif decision.full_house is False:
            dtype = "partial"
        else:
            dtype = "discovery"

        # Normalize to list
        if isinstance(result.sam_polygons, list):
            polys = result.sam_polygons
        else:
            polys = [result.sam_polygons]

        final_polys = []

        for refined_poly in polys:

            final_polys.append(refined_poly)
            all_refined_polys.extend(final_polys)
            images[row.tiff_path] = img


        write_detected_houses(
            building_id=row.id,
            polygons=final_polys,
            detection_type=dtype,
            run_id=RUN_ID,
            tiff_path=row.tiff_path,
            win=result.metadata.get("win", win),
            metadata=result.metadata,
        )

        tree_polys = result.metadata.get("tree_polygons", [])

        write_detected_trees(
            building_id=row.id,
            polygons=tree_polys,
            run_id=RUN_ID,
            tiff_path=row.tiff_path,
            win=result.metadata.get("win", win),
            metadata=result.metadata,
        )

    # ---------------------------------------------
    # Write MLQA results for all pipelines
    # ---------------------------------------------
    
    write_mlqa({
        "building_id": row.id,
        "patch_path": str(ctx.discovery_path or clean_path),
        "house_present": decision.house_present,
        "full_house_present": decision.full_house,
        "error_description": decision.error,
        "errors": decision.errors,
        "inside_pts": result.inside_pts,
        "outside_pts": result.outside_pts,
    })
    
    # Log SAM results
    if result.sam_polygons:
        if isinstance(result.sam_polygons, list):
            print(f"  ✓ {result.pipeline_name}: {len(result.sam_polygons)} building(s) segmented")
        else:
            print(f"  ✓ {result.pipeline_name}: 1 building segmented")
    else:
        print(f"  ✓ {result.pipeline_name}: No SAM segmentation")

print("\nDONE")

# --------------------------------------------------
# GLOBAL DISCOVERY (FIXED: Use full AOI image)
# --------------------------------------------------



if ENABLE_GLOBAL:
    print("\n GLOBAL DISCOVERY START")

    aoi_tiff_path = gdf.iloc[0]["tiff_path"]

    aoi_img_rgb, aoi_transform, aoi_crs = extract_aoi_raster(
        aoi_geom=aoi_gdf.iloc[0]["geom"],
        aoi_crs=aoi_gdf.crs,
        tiff_path=aoi_tiff_path,
    )

    print(f"AOI image shape: {aoi_img_rgb.shape}")

    candidates = run_global_discovery(
        image=aoi_img_rgb,  # Full AOI RGB image
        prompt="roof",
        tile_size=1024,
        overlap=128
    )

    print(f"Total candidates: {len(candidates)}")

    # ----------------------------
    # 1. DEDUPLICATION
    # ----------------------------
    deduped_px = deduplicate_polygons(candidates)

    print(f"After deduplication: {len(deduped_px)}")

    deduped = deduped_px

    # ----------------------------
    # 3. MATCHING
    # ----------------------------
    new_buildings = filter_new_buildings(
        deduped,
        all_refined_polys,
        iou_threshold=0.3
    )

    print(f"New buildings: {len(new_buildings)}")

    # ----------------------------
    # 3. SAVE
    # ----------------------------
    if new_buildings:
        # Use AOI TIFF (safe fallback)
        aoi_tiff_path = gdf.iloc[0]["tiff_path"] if not gdf.empty else "unknown.tif"

        write_detected_houses(
            building_id=-1,
            polygons=new_buildings,
            detection_type="global_discovery",
            run_id=RUN_ID,
            tiff_path=aoi_tiff_path,  # Safe!
            win=None,
            metadata={
                "stage": "global",
                "raw_count": len(candidates),
                "transform": aoi_transform  # ← CRITICAL FIX
            })

# --------------------------------------------------
# Export to FileGDB
# --------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gdb_output = output_dir / f"building_results_{timestamp}.gdb"

export_buildings_to_filegdb(
    engine=engine,
    output_path=str(gdb_output),
    aoi_id=AOI_ID,
    run_id=RUN_ID,
    overwrite=True,
)
