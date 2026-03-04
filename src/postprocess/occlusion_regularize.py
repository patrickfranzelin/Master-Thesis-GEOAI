from __future__ import annotations

import argparse
import os
from typing import Iterable

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.wkt import loads as wkt_loads
from sqlalchemy import create_engine, text

try:
    from buildingregulariser import regularize_geodataframe
except ImportError as exc:
    raise ImportError(
        "buildingregulariser is not installed. Install with: pip install buildingregulariser"
    ) from exc


def get_local_utm_crs(gdf):
    centroid = gdf.union_all().centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = 32700 + zone if centroid.y < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def remove_tree_occlusion(
    roof_poly: Polygon,
    tree_polys: Iterable[Polygon],
    buffer_size: float = 0.00001,
) -> tuple[Polygon, float]:
    tree_polys = list(tree_polys)
    if not tree_polys:
        return roof_poly, 0.0

    tree_union = unary_union(tree_polys)
    occluded_area = roof_poly.intersection(tree_union).area
    occlusion_ratio = occluded_area / roof_poly.area if roof_poly.area else 0.0

    roof_visible = roof_poly.difference(tree_union)
    if roof_visible.is_empty:
        return roof_poly, 1.0

    if isinstance(roof_visible, MultiPolygon):
        roof_visible = max(roof_visible.geoms, key=lambda g: g.area)

    regularized = roof_visible.buffer(buffer_size).buffer(-buffer_size)
    if regularized.is_empty:
        return roof_poly, occlusion_ratio

    if isinstance(regularized, MultiPolygon):
        regularized = max(regularized.geoms, key=lambda g: g.area)

    return regularized, occlusion_ratio
# Add this function anywhere before process_buildings:

def normalize_blob(polygon: Polygon, *, blob_tolerance: float = 0.4) -> Polygon:
    """
    Fix SAM blob artifacts (inward and outward) via morphological opening.
    Erode removes outward bumps, dilate back fills inward dents.
    blob_tolerance in metres (UTM input expected).
    """
    if polygon is None or polygon.is_empty:
        return polygon

    eroded = polygon.buffer(-blob_tolerance)
    if eroded.is_empty:
        return polygon
    if isinstance(eroded, MultiPolygon):
        eroded = max(eroded.geoms, key=lambda g: g.area)

    restored = eroded.buffer(blob_tolerance)
    if restored.is_empty:
        return polygon
    if isinstance(restored, MultiPolygon):
        restored = max(restored.geoms, key=lambda g: g.area)

    # Reject if over-eroded (lost >30% area)
    if restored.area < polygon.area * 0.7:
        return polygon

    return restored


def process_buildings(
    run_id: str,
    source_table: str,
    tree_table: str,
    target_schema: str,
    target_table: str,
    output_conn: str,
    *,
    source_conn: str,
    buffer_size: float,
    include_45_degree: bool,
    include_circles: bool,
    simplify_tolerance: float,
    parallel_threshold: float,
    neighbor_alignment: bool,
    neighbor_search_distance: float,
    neighbor_max_rotation: float,
    replace_target: bool,
    min_area_m2: float = 5.0,
) -> int:

    source_engine = create_engine(source_conn)

    # ST_AsText avoids WKB hex/bytes ambiguity entirely
    sql = text(f"""
        SELECT h.id,
               h.building_id,
               h.run_id,
               h.detection_type,
               h.created_at,
               ST_AsText(h.geom) AS roof_wkt,
               ST_AsText(t.geom) AS tree_wkt
        FROM {source_table} h
        LEFT JOIN {tree_table} t
               ON h.building_id = t.building_id
              AND h.run_id = t.run_id
        WHERE h.run_id = :run_id
    """)

    with source_engine.begin() as conn:
        rows = conn.execute(sql, {"run_id": run_id}).mappings().all()

    if not rows:
        print(f"No source buildings found for run_id={run_id}.")
        return 0

    grouped: dict[tuple[int, str], dict] = {}
    for row in rows:
        key = (row["id"], row["run_id"])
        if key not in grouped:
            grouped[key] = {
                "id":             row["id"],
                "building_id":    row["building_id"],
                "run_id":         row["run_id"],
                "detection_type": row["detection_type"],
                "created_at":     row["created_at"],
                "roof":           wkt_loads(row["roof_wkt"]),
                "trees":          [],
            }
        if row["tree_wkt"] is not None:
            grouped[key]["trees"].append(wkt_loads(row["tree_wkt"]))

    base_records = []
    for entry in grouped.values():
        cleaned_geom, occlusion_ratio = remove_tree_occlusion(
            roof_poly=entry["roof"],
            tree_polys=entry["trees"],
            buffer_size=buffer_size,
        )
        base_records.append({
            "source_id":         entry["id"],
            "building_id":       entry["building_id"],
            "run_id":            entry["run_id"],
            "detection_type":    entry["detection_type"],
            "source_created_at": entry["created_at"],
            "occlusion_ratio":   occlusion_ratio,
            "geometry":          cleaned_geom,
        })

    gdf = gpd.GeoDataFrame(base_records, geometry="geometry", crs="EPSG:4326")
    utm_crs = get_local_utm_crs(gdf)
    gdf_utm = gdf.to_crs(utm_crs)

    # Drop noise fragments smaller than min_area_m2
    gdf_utm = gdf_utm[gdf_utm.geometry.area >= min_area_m2].copy()
    if gdf_utm.empty:
        print("No buildings remaining after area filter.")
        return 0
    gdf_utm["geometry"] = gdf_utm["geometry"].apply(
        lambda g: normalize_blob(g, blob_tolerance=0.4) if g is not None and not g.is_empty else g
    )

    regularized = regularize_geodataframe(
        gdf_utm,
        simplify=True,
        simplify_tolerance=0.15,  # 3× pixel size (0.05m) — removes sub-15cm noise
        parallel_threshold=0.5,  # snaps edges within ~0.5m of parallel — good for 5cm imagery
        allow_45_degree=False,  # disable — informal buildings rarely have true 45° edges
        allow_circles=False,
        diagonal_threshold_reduction=20.0,  # max value — heavily suppresses false diagonal detection
        neighbor_alignment=False,
    )

    # Ensure geometry column is active
    geom_col = "geometry" if "geometry" in regularized.columns else "geom"
    regularized = regularized.rename(columns={geom_col: "geometry"})

    # Keep largest polygon per building_id
    # Keep largest polygon per building_id
    regularized["area_m2"] = regularized["geometry"].apply(lambda g: g.area)
    regularized = (
        regularized
        .sort_values("area_m2", ascending=False)
        .groupby("building_id", as_index=False)
        .first()
        .reset_index(drop=True)
    )

    # Restore CRS after groupby drops it
    regularized = gpd.GeoDataFrame(regularized, geometry="geometry", crs=utm_crs)

    # Compute area in m² while still in UTM, then drop temp column
    regularized["area"] = regularized.geometry.area
    regularized = regularized.drop(columns=["area_m2"])

    # Back to WGS84 for storage
    regularized = regularized.to_crs(4326)



    output_engine = create_engine(output_conn)
    if_exists = "replace" if replace_target else "append"
    regularized.to_postgis(
        name=target_table,
        con=output_engine,
        schema=target_schema,
        if_exists=if_exists,
        index=False,
    )

    print(f"Wrote {len(regularized)} rows to {target_schema}.{target_table} (if_exists={if_exists})")
    return len(regularized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove tree occlusion, regularize footprints, and write to a target PostGIS table."
    )
    parser.add_argument("--run-id", default="a523c6d0-fc1e-412e-a27c-0a861c0c198b")
    parser.add_argument("--source-table",  default="src.detected_house")
    parser.add_argument("--tree-table",    default="src.detected_tree")
    parser.add_argument("--target-schema", default="src")
    parser.add_argument("--target-table",  default="detected_house_regularized")
    parser.add_argument("--replace-target", action="store_true")
    parser.add_argument("--buffer-size",   type=float, default=0.00001)
    parser.add_argument("--parallel-threshold",    type=float, default=1.0)
    parser.add_argument("--simplify-tolerance",    type=float, default=0.4)
    parser.add_argument("--allow-45-degree",       action="store_true")
    parser.add_argument("--allow-circles",         action="store_true")
    parser.add_argument("--neighbor-alignment",    action="store_true")
    parser.add_argument("--neighbor-search-distance", type=float, default=350.0)
    parser.add_argument("--neighbor-max-rotation",    type=float, default=10.0)
    parser.add_argument("--min-area-m2",           type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_conn = os.environ["PG_CONN"]
    output_conn = os.environ.get("OUTPUT_PG_CONN", source_conn)

    process_buildings(
        run_id=args.run_id,
        source_table=args.source_table,
        tree_table=args.tree_table,
        target_schema=args.target_schema,
        target_table=args.target_table,
        output_conn=output_conn,
        source_conn=source_conn,
        buffer_size=args.buffer_size,
        include_45_degree=args.allow_45_degree,
        include_circles=args.allow_circles,
        simplify_tolerance=args.simplify_tolerance,
        parallel_threshold=args.parallel_threshold,
        neighbor_alignment=args.neighbor_alignment,
        neighbor_search_distance=args.neighbor_search_distance,
        neighbor_max_rotation=args.neighbor_max_rotation,
        replace_target=args.replace_target,
        min_area_m2=args.min_area_m2,
    )


if __name__ == "__main__":
    main()
