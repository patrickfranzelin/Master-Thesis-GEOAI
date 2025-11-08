from __future__ import annotations
import os, json, yaml
import geopandas as gpd
from shapely.geometry import Point

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)

def save_json(obj, path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def save_points_to_gpkg(out_path, inside_global, outside_global, poly_id, crs):
    """Save MLLM inside/outside points as GeoPackage."""
    gdf_points = gpd.GeoDataFrame({
        "poly_id": [poly_id] * (len(inside_global) + len(outside_global)),
        "type": (["inside"] * len(inside_global)) + (["outside"] * len(outside_global)),
        "geometry": [Point(p) for p in inside_global + outside_global]
    }, crs=crs)

    gdf_points.to_file(out_path, driver="GPKG", layer="points",
                       mode="a" if os.path.exists(out_path) else "w")
