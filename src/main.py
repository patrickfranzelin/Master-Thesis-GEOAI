import pyproj
pyproj.datadir.set_data_dir(r"C:\Users\franz\miniconda3\envs\geoai\Library\share\proj")

from pathlib import Path
import cv2

from src.image_utils import add_center_star, add_grid_overlay
from src.patch_generator import load_gdb_polygons, extract_patch_from_gdb


# -------- paths --------

tif_path = r"C:\Users\franz\Downloads\5e53f6ea906c590005ecc5ba.tif"

gdb_path = r"C:\Users\franz\OneDrive\Dokumente\ArcGIS\Projects\QA_Assesing_Google_Footprints\QA_Assesing_Google_Footprints.gdb"
layer_name = "google_buildings_Clip"

output_dir = Path(__file__).resolve().parent.parent / "gdb_results"
output_dir.mkdir(exist_ok=True)


# -------- run --------

gdf = load_gdb_polygons(gdb_path, layer_name, 50)

for idx, row in gdf.iterrows():

    img, _, _ = extract_patch_from_gdb(
        row.geometry,
        gdf.crs,
        tif_path,
        buffer_m=20
    )

    img = add_center_star(img)
    img = add_grid_overlay(img)

    cv2.imwrite(
        str(output_dir / f"bld_{idx:03d}.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )

    print(f"✅ bld_{idx:03d}")
