from pathlib import Path
import cv2
import geopandas as gpd
from sqlalchemy import create_engine
import os

from src.core.context import PipelineContext
from src.mlqa.decision import decide
from src.mlqa.mlqa_client import MLQAParseError
from src.pipelines.router import route
from src.patches.extractor import extract_patch
from src.patches.create_patch_output import create_patch_outputs
from src.db.writer import write_mlqa
from src.db.writer import write_detected_houses

# --------------------------------------------------
# Paths
# --------------------------------------------------

output_dir = Path("../outputs/db_results")
sam_dir = output_dir / "sam"
raw_dir = output_dir / "raw"
clean_dir = output_dir / "clean"
debug_dir = output_dir / "debug"

for d in [sam_dir, raw_dir, clean_dir, debug_dir]:
    d.mkdir(parents=True, exist_ok=True)

out_dirs = {
    "raw": raw_dir,
    "clean": clean_dir,
    "debug": debug_dir,
}

# --------------------------------------------------
# Database
# --------------------------------------------------

engine = create_engine(os.environ["PG_CONN"])
AOI_ID = 1

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

for _, row in gdf.iterrows():

    print(f"\nProcessing building {row.id}")

    # ---------------------------------------------
    # Patch extraction
    # ---------------------------------------------

    img, poly_px = extract_patch(row.geom, gdf.crs, row.tiff_path, context=1.5)
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
        if isinstance(result.sam_polygons, list):
            polys = result.sam_polygons
        else:
            polys = [result.sam_polygons]

        # Determine detection type
        if decision.full_house is True:
            dtype = "full"
        elif decision.full_house is False:
            dtype = "partial"
        else:
            dtype = "discovery"

        write_detected_houses(
            building_id=row.id,
            polygons=polys,
            detection_type=dtype,
        )

    # ---------------------------------------------
    # Write MLQA results for all pipelines (fixes Bug 4)
    # ---------------------------------------------
    
    write_mlqa({
        "building_id": row.id,
        "patch_path": str(ctx.discovery_path or clean_path),
        "house_present": decision.house_present,
        "full_house_present": decision.full_house,
        "error_description": decision.error,
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
