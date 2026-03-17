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


import math

def fix_tree_dents(
    poly: Polygon,
    tree_union: Polygon,
    max_dent_area=50,
    proximity_dist=2.0,
    direction_check=True,
):
    if poly.is_empty:
        return poly

    hull = poly.convex_hull
    dents = hull.difference(poly)

    if dents.is_empty:
        return poly

    if dents.geom_type == "Polygon":
        dents = [dents]
    else:
        dents = list(dents.geoms)

    cleaned = poly

    for dent in dents:
        # --- 1. ignore large structures (not tree dents)
        if dent.area > max_dent_area:
            continue

        # --- 2. must be near a tree
        if dent.distance(tree_union) > proximity_dist:
            continue

        # --- 3. OPTIONAL: direction check (VERY IMPORTANT)
        if direction_check:
            dent_center = dent.centroid

            # find nearest point on tree
            nearest_tree_point = tree_union.exterior.interpolate(
                tree_union.exterior.project(dent_center)
            )

            # vector dent → tree
            dx = nearest_tree_point.x - dent_center.x
            dy = nearest_tree_point.y - dent_center.y

            # vector dent → polygon center
            poly_center = poly.centroid
            px = poly_center.x - dent_center.x
            py = poly_center.y - dent_center.y

            # dot product: should be opposite directions
            dot = dx * px + dy * py

            # if positive → dent points inward (wrong direction)
            if dot > 0:
                continue

        # --- 4. fill dent locally
        cleaned = cleaned.union(dent)

    return cleaned



def merge_polygons_with_tracking(gdf, buffer_dist=0.3):
    """
    Merge polygons that overlap or are very close.
    Tracks original building_ids.
    """

    geoms = list(gdf.geometry)
    records = gdf.to_dict("records")

    used = [False] * len(geoms)
    merged_rows = []

    for i, geom in enumerate(geoms):
        if used[i]:
            continue

        group_geoms = [geom]
        group_ids = [records[i]["building_id"]]
        group_records = [records[i]]

        used[i] = True

        for j in range(i + 1, len(geoms)):
            if used[j]:
                continue

            #  KEY CONDITION (overlap OR very close)
            if geom.intersects(geoms[j]) or geom.distance(geoms[j]) < buffer_dist:
                group_geoms.append(geoms[j])
                group_ids.append(records[j]["building_id"])
                group_records.append(records[j])
                used[j] = True

        #  merge robustly
        buffered = [g.buffer(buffer_dist) for g in group_geoms]
        merged_geom = unary_union(buffered).buffer(-buffer_dist)

        if merged_geom.is_empty:
            continue

        if isinstance(merged_geom, MultiPolygon):
            merged_geom = max(merged_geom.geoms, key=lambda g: g.area)

        # choose representative row (largest polygon)
        best = max(group_records, key=lambda r: r["geometry"].area)

        row = best.copy()
        row["geometry"] = merged_geom

        #  store provenance
        row["merged_from"] = list(set(group_ids))

        merged_rows.append(row)

    return gpd.GeoDataFrame(merged_rows, crs=gdf.crs)

from shapely.ops import unary_union

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

    # -------------------------
    # GROUP DATA
    # -------------------------
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

    # -------------------------
    # BUILD BASE GDF (NO OCCLUSION REMOVAL)
    # -------------------------
    base_records = []
    for entry in grouped.values():
        base_records.append({
            "source_id":         entry["id"],
            "building_id":       entry["building_id"],
            "run_id":            entry["run_id"],
            "detection_type":    entry["detection_type"],
            "source_created_at": entry["created_at"],
            "occlusion_ratio":   0.0,                 # removed logic
            "geometry":          entry["roof"],       # KEEP ORIGINAL
        })

    gdf = gpd.GeoDataFrame(base_records, geometry="geometry", crs="EPSG:4326")

    # -------------------------
    # PROJECT TO UTM
    # -------------------------
    utm_crs = get_local_utm_crs(gdf)
    gdf_utm = gdf.to_crs(utm_crs)

    # -------------------------
    # MERGE BUILDINGS
    # -------------------------
    gdf_utm = merge_polygons_with_tracking(gdf_utm, buffer_dist=0.3)

    # -------------------------
    # BUILD TREE UNION (FIXED)
    # -------------------------
    all_trees = [
        t
        for entry in grouped.values()
        for t in entry["trees"]
    ]

    if all_trees:
        raw_tree_union = unary_union(all_trees)
        tree_union = raw_tree_union.buffer(0.5)   # small buffer ONLY
    else:
        tree_union = None

    # -------------------------
    # FIX TREE DENTS (MAIN STEP)
    # -------------------------
    if tree_union:
        gdf_utm["geometry"] = gdf_utm["geometry"].apply(
            lambda g: fix_tree_dents(
                g,
                tree_union,
                max_dent_area=60,
                proximity_dist=2.5,
            )
        )

    # -------------------------
    # REGULARIZATION (UNCHANGED)
    # -------------------------
    regularized = regularize_geodataframe(
        gdf_utm,
        simplify=True,
        parallel_threshold=1,
        simplify_tolerance=0.5,
        allow_45_degree=True,
        allow_circles=True,
        circle_threshold=0.9,
        neighbor_alignment=False
    )

    geom_col = "geometry" if "geometry" in regularized.columns else "geom"
    regularized = regularized.rename(columns={geom_col: "geometry"})

    regularized["area_m2"] = regularized["geometry"].apply(lambda g: g.area)

    regularized = gpd.GeoDataFrame(regularized, geometry="geometry", crs=utm_crs)

    regularized["area"] = regularized.geometry.area
    regularized = regularized.drop(columns=["area_m2"])

    regularized = regularized.to_crs(4326)

    # -------------------------
    # WRITE OUTPUT (UNCHANGED)
    # -------------------------
    output_engine = create_engine(output_conn)
    if_exists = "replace" if replace_target else "append"

    regularized = regularized.rename(columns={"geometry": "geom"})
    regularized = gpd.GeoDataFrame(regularized, geometry="geom", crs=4326)

    if "merged_from" not in regularized.columns:
        regularized["merged_from"] = None

    if "source_created_at" in regularized.columns:
        regularized["created_at"] = regularized["source_created_at"]
    else:
        regularized["created_at"] = None

    regularized = regularized[
        [
            "building_id",
            "run_id",
            "created_at",
            "area",
            "merged_from",
            "geom",
        ]
    ]

    import json
    regularized["merged_from"] = regularized["merged_from"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else None
    )

    regularized.to_postgis(
        name=target_table,
        con=output_engine,
        schema=target_schema,
        if_exists=if_exists,
        index=False,
        dtype={"geom": "Geometry(Polygon, 4326)"}
    )

    print(f"Wrote {len(regularized)} rows to {target_schema}.{target_table} (if_exists={if_exists})")
    return len(regularized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove tree occlusion, regularize footprints, and write to a target PostGIS table."
    )
    parser.add_argument("--run-id", default="1a9332c0-bc57-45c3-90f7-76dbef772368")
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
