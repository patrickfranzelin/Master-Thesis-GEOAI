from __future__ import annotations
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ..geo.projection import to_meters, back_from_meters
from pyproj import CRS

class PolygonPostProcessor:
    def __init__(self, cfg: dict, src_crs):
        self.cfg = cfg
        self.src_crs = src_crs

    def _clean_polygon(self, poly_m):
        buf, dbuf = self.cfg["clean_buffer_m"], self.cfg["clean_debuffer_m"]
        try:
            if buf > 0:  poly_m = poly_m.buffer(buf)
            if dbuf > 0: poly_m = poly_m.buffer(-dbuf)
            if not poly_m.is_valid: poly_m = poly_m.buffer(0)
        except Exception: pass
        return poly_m

    def filter_and_clean(self, polys: list[Polygon], attrs: list[dict]) -> list[Polygon]:
        out: list[Polygon] = []
        for p, a in zip(polys, attrs):
            pm, mode = to_meters(p, self.src_crs)
            area = pm.area
            if not (self.cfg["min_area_m2"] <= area <= self.cfg["max_area_m2"]):
                continue
            pm = self._clean_polygon(pm)
            p2 = back_from_meters(pm, self.src_crs, mode)
            if p2.is_valid and not p2.is_empty:
                out.append(p2)
        return out

    def merge_close(self, polys: list[Polygon]) -> list[Polygon]:
        if not polys: return []
        pmode = []
        for p in polys:
            pm, mode = to_meters(p, self.src_crs)
            pmode.append((pm, mode))
        merged = unary_union([pm.buffer(self.cfg["merge_gap_m"]/2) for pm,_ in pmode]).buffer(-self.cfg["merge_gap_m"]/2)
        res: list[Polygon] = []
        geoms = (list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]) if not merged.is_empty else []
        for gm in geoms:
            gm = self._clean_polygon(gm)
            if gm.is_empty: continue
            # back to src
            res.append(back_from_meters(gm, self.src_crs, "used_3857"))  # if src was proj, transform is identity
        return res
