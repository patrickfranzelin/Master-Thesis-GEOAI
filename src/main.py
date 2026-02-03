import os

os.environ["PROJ_LIB"] = r"C:\Users\franz\miniconda3\envs\geoai-qa\Library\share\proj"
RUNPOD_ID = os.environ["RUNPOD_ID"]
PG_CONN = os.environ["PG_CONN"]

from pathlib import Path
import cv2

from src.db_loader import load_buildings
from src.patch_generator import extract_patch
from src.image_utils import add_center_star, add_grid_overlay
from src.mlqa.mlqa_client import analyze_patch
from src.image_utils import add_polygon_overlay
from src.mlqa.mlqa_writer import write_mlqa




output_dir = Path("../outputs/db_results")
output_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# Load buildings from Postgres
# --------------------------------------------------

gdf = load_buildings(limit=500)  # remove limit later

# --------------------------------------------------
# Extract patches
# --------------------------------------------------

results = []

clean_dir = output_dir / "clean"
debug_dir = output_dir / "debug"

clean_dir.mkdir(exist_ok=True)
debug_dir.mkdir(exist_ok=True)

for idx, row in gdf.iterrows():

    img, poly_px = extract_patch(row.geom, gdf.crs, row.tiff_path)

    # -----------------------------
    # CLEAN PATCH (polygon only)
    # -----------------------------

    clean_img = add_polygon_overlay(img.copy(), poly_px)

    clean_path = clean_dir / f"bld_{row.id:07d}_clean.png"
    cv2.imwrite(str(clean_path), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
    print(f"Saved {clean_path.name} + debug")
    # -----------------------------
    # DEBUG PATCH (polygon + star + grid)
    # -----------------------------

    debug_img = add_polygon_overlay(img.copy(), poly_px)
    debug_img = add_center_star(debug_img)
    debug_img = add_grid_overlay(debug_img)

    #debug_path = debug_dir / f"bld_{row.id:07d}_debug.png"
    #cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    #print(f"Saved {clean_path.name} + debug")

    # -----------------------------
    # MLQA (use CLEAN image!)
    # -----------------------------

    qa = analyze_patch(clean_path)

    print("QA:", qa)

    record = {
        "building_id": int(row.id),
        "error_description": qa["error_description"],
    }

    write_mlqa(record)



print("\nDONE")


