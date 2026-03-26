import os
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

import cv2
import numpy as np
import rasterio

from shapely import wkt
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from src.patches.extractor import extract_patch
from src.utils.rendering import add_polygon_overlay


# ----------------------------
# CONFIG
# ----------------------------
OUT_DIR = Path("data")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(os.environ["PG_CONN"])

LIMIT = 1000
RUN_ID = "97cd6744-3b24-4e2c-9ad7-fe65a2a32bbd"  # e.g. 42


# ----------------------------
# HELPERS
# ----------------------------
def to_geom(w):
    if w is None:
        return None
    try:
        g = wkt.loads(w)
        return g if not g.is_empty else None
    except Exception:
        return None


def debug_geom(name, geom):
    if geom is None:
        print(f"  {name}: NONE")
    else:
        print(f"  {name}: OK | area={geom.area:.2f} | bounds={geom.bounds}")


# ----------------------------
# SQL
# ----------------------------
run_filter_d = f"AND d.run_id = {RUN_ID}" if RUN_ID else ""
run_filter_r = f"AND r.run_id = {RUN_ID}" if RUN_ID else ""

sql = """
SELECT 
    b.id AS building_id,
    b.tiff_path,

    ST_AsText(b.geom) AS original_geom,
    ST_AsText(d.geom) AS sam_geom,
    ST_AsText(r.geom) AS post_geom,

    d.id AS sam_id,
    d.run_id

FROM src.buildings b

JOIN src.detected_house d
    ON d.building_id = b.id
    AND (%(run_id)s IS NULL OR d.run_id = %(run_id)s)

LEFT JOIN src.detected_house_regularized r
    ON r.building_id = b.id
    AND (%(run_id)s IS NULL OR r.run_id = %(run_id)s)

WHERE d.geom IS NOT NULL

LIMIT %(limit)s
"""

# ----------------------------
# LOAD
# ----------------------------
df = pd.read_sql(
    sql,
    engine,
    params={
        "run_id": RUN_ID,
        "limit": LIMIT
    }
)
print(f"Loaded {len(df)} rows")


# ----------------------------
# PROCESS
# ----------------------------
samples = []

for _, row in df.iterrows():

    bid = row.building_id
    sam_id = row.sam_id

    print("\n============================")
    print(f"BUILDING {bid} | SAM {sam_id}")

    try:
        # ----------------------------
        # GEOMS
        # ----------------------------
        original_geom = to_geom(row.original_geom)
        sam_geom = to_geom(row.sam_geom)
        post_geom = to_geom(row.post_geom)

        if original_geom is None or sam_geom is None:
            print("  -> skip (missing geom)")
            continue

        debug_geom("original", original_geom)
        debug_geom("sam", sam_geom)
        debug_geom("post", post_geom)

        # ----------------------------
        # CENTERING
        # ----------------------------
        if post_geom:
            center_geom = post_geom
            print("  → centered on POST")
        else:
            center_geom = sam_geom
            print("  → centered on SAM")

        # ----------------------------
        # PATCH
        # ----------------------------
        img, _, win = extract_patch(
            center_geom,
            "EPSG:4326",
            row.tiff_path,
            context=1.5
        )

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ----------------------------
        # TRANSFORM SETUP
        # ----------------------------
        with rasterio.open(row.tiff_path) as src:
            raster_crs = src.crs
            affine = src.window_transform(win)

        inv = ~affine

        h, w = img.shape[:2]
        sx = w / win.width
        sy = h / win.height

        to_raster = Transformer.from_crs(
            "EPSG:4326",
            raster_crs,
            always_xy=True
        ).transform

        def world_to_pixel(geom, name):
            if geom is None:
                return None

            try:
                g = shp_transform(to_raster, geom)
                g = shp_transform(lambda x, y: inv * (x, y), g)
                g = shp_transform(lambda x, y: (x * sx, y * sy), g)

                minx, miny, maxx, maxy = g.bounds
                print(f"  {name} px bounds: {(round(minx,1), round(miny,1), round(maxx,1), round(maxy,1))}")

                if maxx < 0 or maxy < 0 or minx > w or miny > h:
                    print(f"  ⚠️ {name} OUTSIDE IMAGE")

                return g

            except Exception as e:
                print(f"  ❌ transform failed ({name}): {e}")
                return None

        # ----------------------------
        # TRANSFORM
        # ----------------------------
        original_px = world_to_pixel(original_geom, "original")
        sam_px = world_to_pixel(sam_geom, "sam")
        post_px = world_to_pixel(post_geom, "post") if post_geom else None

        # ----------------------------
        # RENDER
        # ----------------------------
        img_original = img.copy()
        if original_px:
            img_original = add_polygon_overlay(img_original, original_px, (0, 255, 255))

        img_sam = img.copy()
        if sam_px:
            img_sam = add_polygon_overlay(img_sam, sam_px, (0, 0, 255))

        img_post = img.copy()
        if post_px:
            img_post = add_polygon_overlay(img_post, post_px, (0, 200, 0))

        # ----------------------------
        # COMBINE
        # ----------------------------
        divider = np.ones((img.shape[0], 6, 3), dtype=np.uint8) * 220

        combined = np.hstack([
            img_original,
            divider,
            img_sam,
            divider,
            img_post
        ])

        # ----------------------------
        # SAVE IMAGE
        # ----------------------------
        out_name = f"{bid}_run{row.run_id}_sam{sam_id}.png"
        out_path = IMG_DIR / out_name

        cv2.imwrite(str(out_path), combined)

        # ----------------------------
        # STORE META
        # ----------------------------
        samples.append({
            "building_id": int(bid),
            "sam_id": int(sam_id),
            "run_id": str(row.run_id) if row.run_id else None,
            "image": f"data/images/{out_name}",
            "has_post": post_geom is not None
        })

    except Exception as e:
        print(f" FAILED {bid}-{sam_id}: {e}")
        continue


# ----------------------------
# SAVE JSON
# ----------------------------
with open(OUT_DIR / "samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print("\nDONE")