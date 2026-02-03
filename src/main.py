import os

from shapely.geometry import box

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
from src.mlqa.point_client import analyze_points




output_dir = Path("../outputs/db_results")
output_dir.mkdir(exist_ok=True)
points_dir = output_dir / "points"
points_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# Load buildings from Postgres
# --------------------------------------------------
AOI_BBOX = (
    2680000,  # xmin
    1200000,  # ymin
    2682000,  # xmax
    1202000   # ymax
)
aoi_geom = box(*AOI_BBOX)
gdf = load_buildings(limit=500)  # remove limit later
gdf = gdf[gdf.intersects(aoi_geom)]
print(f"Buildings after AOI filter: {len(gdf)}")


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

    # -----------------------------
    # DEBUG PATCH (polygon + star + grid)
    # -----------------------------

    debug_img = add_polygon_overlay(img.copy(), poly_px)
    debug_img = add_center_star(debug_img)
    debug_img = add_grid_overlay(debug_img)

    debug_path = debug_dir / f"bld_{row.id:07d}_debug.png"
    cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    print(f"Saved {clean_path.name} + debug")

    # -----------------------------
    # MLQA (clean image)
    # -----------------------------

    qa = analyze_patch(clean_path)
    print("QA:", qa)

    inside_pts = []
    outside_pts = []

    if qa["house_present"]:

        points = analyze_points(debug_path)
        print("Points:", points)

        inside_pts = points.get("inside", [])
        outside_pts = points.get("outside", [])

        # -----------------------------
        # Draw points + save overlay
        # -----------------------------

        overlay = debug_img.copy()

        h, w = overlay.shape[:2]

        for pt in inside_pts:
            if isinstance(pt, list) and len(pt) == 2:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)

        for pt in outside_pts:
            if isinstance(pt, list) and len(pt) == 2:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(overlay, (x, y), 12, (0, 0, 255), -1)

        points_path = points_dir / f"bld_{row.id:07d}_points.png"
        cv2.imwrite(str(points_path), overlay)

        print(f"Saved points overlay: {points_path.name}")

    record = {
        "building_id": int(row.id),
        "house_present": qa["house_present"],
        "error_description": qa["error_description"],
        "inside_pts": inside_pts,
        "outside_pts": outside_pts
    }

    write_mlqa(record)




print("\nDONE")


