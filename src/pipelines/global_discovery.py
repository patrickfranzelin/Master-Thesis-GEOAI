from src.sam.tile_detection import detect_polygons
from shapely.ops import unary_union
from shapely.validation import make_valid
from pathlib import Path


def ensure_polygon_list(geom):
    """Convert MultiPolygon/GeometryCollection → list of Polygons."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if g.geom_type == "Polygon"]
    return []


def run_global_discovery(image, prompt, tile_size, overlap):
    debug_dir = Path("../outputs/db_results/debug/global_tiles")
    debug_dir.mkdir(exist_ok=True, parents=True)

    print("[GLOBAL] Running tiled SAM detection")

    polys = detect_polygons(
        image=image,
        prompt=prompt,
        image_patch_dir=debug_dir,
        tile_size=tile_size,
        overlap=overlap
    )

    if not polys:
        print("[GLOBAL] No raw polygons found")
        return []

    print(f"[GLOBAL] Raw polygons: {len(polys)}")

    # 1. Repair + filter invalid/small
    valid_polys = []
    for i, p in enumerate(polys):
        if p.is_valid and p.area > 100:
            try:
                valid_polys.append(make_valid(p))
            except:
                continue  # Skip unrepairable
        elif p.area <= 100:
            print(f"[GLOBAL] Skipping tiny poly {i}: {p.area:.0f}px²")

    print(f"[GLOBAL] Valid polys after repair: {len(valid_polys)}")

    if len(valid_polys) == 0:
        return []
    if len(valid_polys) == 1:
        polys = [valid_polys[0]]
    else:
        # 2. Safe iterative union (avoids unary_union crash)
        merged = valid_polys[0]
        for p in valid_polys[1:]:
            try:
                merged = merged.union(p)
            except Exception as e:
                print(f"[GLOBAL] Union skip: {e}")
                continue

        # 3. Extract final polygons
        polys = ensure_polygon_list(merged)

    # 4. Final filter
    polys = [p for p in polys if p.area > 500]

    print(f"[GLOBAL] Final polygons: {len(polys)}")
    return polys
