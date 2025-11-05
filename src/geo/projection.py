from __future__ import annotations
from typing import Tuple
from shapely.ops import transform as shp_transform
from shapely.geometry import base
from pyproj import CRS, Transformer

def is_meter_crs(crs_obj: CRS) -> bool:
    try:
        return crs_obj.is_projected and any("metre" in ai.unit_name.lower() for ai in crs_obj.axis_info)
    except Exception:
        return False

def to_meters(geom: base.BaseGeometry, src_crs) -> Tuple[base.BaseGeometry, str]:
    src = CRS.from_user_input(src_crs)
    if is_meter_crs(src):
        return geom, "src_is_m"
    dst = CRS.from_epsg(3857)
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return shp_transform(tr.transform, geom), "used_3857"

def back_from_meters(geom: base.BaseGeometry, src_crs, mode: str):
    if mode == "src_is_m":
        return geom
    back = Transformer.from_crs(3857, CRS.from_user_input(src_crs), always_xy=True)
    return shp_transform(back.transform, geom)
