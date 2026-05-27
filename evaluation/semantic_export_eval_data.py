import json
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely import wkt
from shapely.ops import transform as shp_transform
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.patches.extractor import extract_patch
from src.utils.rendering import add_polygon_overlay


OUT_DIR = Path(__file__).resolve().parent / "semantic_data"
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(os.environ["PG_CONN"])

SCHEMA = os.environ.get("EVAL_SCHEMA", "src_google")
LIMIT = int(os.environ.get("SEMANTIC_EVAL_LIMIT", "1500"))
PER_COUNTRY = int(os.environ.get("SEMANTIC_EVAL_PER_COUNTRY", "50"))

COUNTRY_CASE = """
case
  when lower(b.tiff_path) like '%mozambique%' then 'Mozambique'
  when lower(b.tiff_path) like '%mexico%' then 'Mexico'
  when lower(b.tiff_path) like '%nepal2%' then 'Nepal2'
  when lower(b.tiff_path) like '%nepal%' then 'Nepal'
  when lower(b.tiff_path) like '%niger%' then 'Niger'
  when lower(b.tiff_path) like '%bangladesh%' then 'Bangladesh'
  when lower(b.tiff_path) like '%liberia%' then 'Liberia'
  else 'Unknown'
end
"""
INCLUDED_COUNTRIES = "('Liberia', 'Mexico', 'Mozambique', 'Nepal', 'Niger')"


def to_geom(value):
    if value is None:
        return None
    try:
        geom = wkt.loads(value)
        return geom if not geom.is_empty else None
    except Exception:
        return None


sql = text(
    f"""
    with mlqa_latest as (
      select distinct on (m.building_id)
        m.building_id,
        m.error_description,
        m.errors,
        m.patch_path,
        m.house_present,
        m.analyzed_at
      from {SCHEMA}.building_mlqa m
      where nullif(trim(m.error_description), '') is not null
        and m.error_description not like 'MLQA_PARSE_ERROR:%'
        and lower(trim(m.error_description)) <> 'no building detected'
        and coalesce(m.house_present, true) = true
      order by m.building_id, m.analyzed_at desc
    ), candidates as (
      select
        b.id as building_id,
        {COUNTRY_CASE} as country,
        b.tiff_path,
        ST_AsText(b.geom) as original_geom,
        m.error_description,
        m.errors,
        m.patch_path,
        row_number() over (
          partition by {COUNTRY_CASE}
          order by b.id
        ) as country_rank
      from {SCHEMA}.buildings b
      join mlqa_latest m on m.building_id = b.id
      where {COUNTRY_CASE} in {INCLUDED_COUNTRIES}
    )
    select
      building_id,
      country,
      tiff_path,
      original_geom,
      error_description,
      errors,
      patch_path
    from candidates
    where country_rank <= :per_country
    order by country, country_rank
    limit :limit
    """
)

df = pd.read_sql(sql, engine, params={"limit": LIMIT, "per_country": PER_COUNTRY})
print(f"Loaded {len(df)} semantic-description rows from {SCHEMA}")
if not df.empty:
    print(df["country"].value_counts().sort_index().to_string())

samples = []

for _, row in df.iterrows():
    building_id = int(row.building_id)
    print(f"Building {building_id}")

    try:
        original_geom = to_geom(row.original_geom)
        if original_geom is None:
            print("  skip: missing original geometry")
            continue

        img, _, win = extract_patch(
            original_geom,
            "EPSG:4326",
            row.tiff_path,
            context=1.5,
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with rasterio.open(row.tiff_path) as src:
            raster_crs = src.crs
            affine = src.window_transform(win)

        inv = ~affine
        h, w = img.shape[:2]
        sx = w / win.width
        sy = h / win.height
        to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True).transform

        geom_px = shp_transform(to_raster, original_geom)
        geom_px = shp_transform(lambda x, y: inv * (x, y), geom_px)
        geom_px = shp_transform(lambda x, y: (x * sx, y * sy), geom_px)

        overlay = add_polygon_overlay(img.copy(), geom_px, (0, 220, 0))

        out_name = f"{building_id}_semantic.png"
        out_path = IMG_DIR / out_name
        cv2.imwrite(str(out_path), overlay)

        samples.append(
            {
                "building_id": building_id,
                "country": str(row.country),
                "image": f"semantic_data/images/{out_name}",
                "error_description": str(row.error_description),
                "mlqa_errors": row.errors,
                "patch_path": row.patch_path,
            }
        )
    except Exception as exc:
        print(f"  failed: {exc}")

with (OUT_DIR / "semantic_samples.json").open("w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(samples)} samples to {OUT_DIR / 'semantic_samples.json'}")
