from __future__ import annotations

import argparse
import os
from shapely.ops import transform
import pyproj
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import MultiPolygon
from shapely.wkt import loads as wkt_loads
from sqlalchemy import create_engine, text

from shapely.ops import unary_union

# --- NEW modular imports ---
from postprocess.occlusion.tree_union import build_tree_union
from postprocess.occlusion.pipeline import apply_tree_occlusion_fix

try:
    from buildingregulariser import regularize_geodataframe
except ImportError as exc:
    raise ImportError(
        "buildingregulariser is not installed. Install with: pip install buildingregulariser"
    ) from exc


# ---------------------------------------------------------
# CRS
# ---------------------------------------------------------
def get_local_utm_crs(gdf):
    centroid = gdf.union_all().centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = 32700 + zone if centroid.y < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------
def merge_polygons_with_tracking(gdf, buffer_dist=0.3):

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


            intersection = geom.intersection(geoms[j]).area
            union = geom.area + geoms[j].area - intersection

            iou = intersection / union if union > 0 else 0

            touching = geom.touches(geoms[j])
            contains = geom.contains(geoms[j]) or geoms[j].contains(geom)

            if (
                    iou > 0.2  # strong overlap → same building
                    or contains  # one inside another
            ):
                group_geoms.append(geoms[j])
                used[j] = True

        buffered = [g.buffer(buffer_dist) for g in group_geoms]
        merged_geom = unary_union(buffered).buffer(-buffer_dist)

        if merged_geom.is_empty:
            continue

        if isinstance(merged_geom, MultiPolygon):
            merged_geom = max(merged_geom.geoms, key=lambda g: g.area)

        best = max(group_records, key=lambda r: r["geometry"].area)

        row = best.copy()
        row["geometry"] = merged_geom
        row["merged_from"] = list(set(group_ids))

        merged_rows.append(row)

    return gpd.GeoDataFrame(merged_rows, crs=gdf.crs)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def process_buildings(
    run_id: str,
    source_table: str,
    tree_table: str,
    target_schema: str,
    target_table: str,
    output_conn: str,
    *,
    source_conn: str,
    include_45_degree: bool,
    include_circles: bool,
    simplify_tolerance: float,
    parallel_threshold: float,
    neighbor_alignment: bool,

    replace_target: bool,
) -> int:

    source_engine = create_engine(source_conn)

    sql = text(f"""
        SELECT 
            h.id,
            h.building_id,
            h.run_id,
            h.detection_type,
            h.created_at,
            ST_AsText(h.geom) AS roof_wkt,
            ST_AsText(t.geom) AS tree_wkt
        FROM {source_table} h
        LEFT JOIN {tree_table} t
          ON h.run_id = t.run_id
         AND ST_DWithin(
                ST_Transform(h.geom, 3857),
                ST_Transform(t.geom, 3857),
                2.0
             )
        WHERE h.run_id = :run_id
    """)

    with source_engine.begin() as conn:
        rows = conn.execute(sql, {"run_id": run_id}).mappings().all()

    if not rows:
        print(f"No source buildings found for run_id={run_id}.")
        return 0

    # ---------------------------------------------------------
    # GROUP DATA
    # ---------------------------------------------------------
    grouped: dict[tuple[int, str], dict] = {}

    for row in rows:
        key = (row["id"], row["run_id"])

        if key not in grouped:
            grouped[key] = {
                "id": row["id"],
                "building_id": row["building_id"],
                "run_id": row["run_id"],
                "detection_type": row["detection_type"],
                "created_at": row["created_at"],
                "roof": wkt_loads(row["roof_wkt"]),
                "trees": [],
            }

        if row["tree_wkt"] is not None:
            grouped[key]["trees"].append(wkt_loads(row["tree_wkt"]))

    # ---------------------------------------------------------
    # BASE GDF
    # ---------------------------------------------------------
    base_records = []

    for entry in grouped.values():
        base_records.append({
            "source_id": entry["id"],
            "building_id": entry["building_id"],
            "run_id": entry["run_id"],
            "detection_type": entry["detection_type"],
            "source_created_at": entry["created_at"],
            "occlusion_ratio": 0.0,
            "geometry": entry["roof"],
        })

    gdf = gpd.GeoDataFrame(base_records, geometry="geometry", crs="EPSG:4326")

    # ---------------------------------------------------------
    # PROJECT
    # ---------------------------------------------------------
    utm_crs = get_local_utm_crs(gdf)
    gdf_utm = gdf.to_crs(utm_crs)


    project = pyproj.Transformer.from_crs(
        "EPSG:4326",
        utm_crs,
        always_xy=True
    ).transform

    #  PROJECT TREES
    for entry in grouped.values():
        entry["trees"] = [
            transform(project, t) for t in entry["trees"]
        ]

    # ---------------------------------------------------------
    # MERGE
    # ---------------------------------------------------------
    gdf_utm = merge_polygons_with_tracking(gdf_utm, buffer_dist=0.3)

    # ---------------------------------------------------------
    # TREE OCCLUSION FIX (NEW CLEAN MODULE)
    # ---------------------------------------------------------
    tree_union = build_tree_union(grouped)

    print("Tree union is None:", tree_union is None)
    if tree_union:
        print("Tree union area:", tree_union.area)

    gdf_utm = apply_tree_occlusion_fix(gdf_utm, tree_union)

    # ---------------------------------------------------------
    # REGULARIZATION
    # ---------------------------------------------------------
    regularized = regularize_geodataframe(
        gdf_utm,
        simplify=True,
        parallel_threshold=2,
        simplify_tolerance=0.4,
        allow_45_degree=False,
        allow_circles=True,
        circle_threshold=0.9,
        neighbor_alignment=False
    )

    geom_col = "geometry" if "geometry" in regularized.columns else "geom"
    regularized = regularized.rename(columns={geom_col: "geometry"})

    regularized = gpd.GeoDataFrame(regularized, geometry="geometry", crs=utm_crs)

    regularized["area"] = regularized.geometry.area

    regularized = regularized.to_crs(4326)
    # ---------------------------------------------------------
    # SKIP REGULARIZATION (DEBUG TREE OCCLUSION ONLY)
    # ---------------------------------------------------------
    #regularized = gdf_utm.copy()

    #regularized["area"] = regularized.geometry.area
    #regularized = regularized.to_crs(4326)

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------
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

    def ensure_single_polygon(g):
        if g is None:
            return None

        # fix invalid geometries
        if not g.is_valid:
            g = g.buffer(0)

        # MultiPolygon → keep largest
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)

        return g

    regularized["geom"] = regularized["geom"].apply(ensure_single_polygon)

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


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tree occlusion + building regularization pipeline"
    )

    parser.add_argument("--run-id", default="97cd6744-3b24-4e2c-9ad7-fe65a2a32bbd")
    parser.add_argument("--source-table", default="src.detected_house")
    parser.add_argument("--tree-table", default="src.detected_tree")
    parser.add_argument("--target-schema", default="src")
    parser.add_argument("--target-table", default="detected_house_regularized")
    parser.add_argument("--replace-target", action="store_true")

    parser.add_argument("--parallel-threshold", type=float, default=1.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.5)

    parser.add_argument("--allow-45-degree", action="store_true")
    parser.add_argument("--allow-circles", action="store_true")
    parser.add_argument("--neighbor-alignment", action="store_true")

    parser.add_argument("--neighbor-search-distance", type=float, default=350.0)
    parser.add_argument("--neighbor-max-rotation", type=float, default=10.0)

    parser.add_argument("--min-area-m2", type=float, default=5.0)

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
        include_45_degree=args.allow_45_degree,
        include_circles=args.allow_circles,
        simplify_tolerance=args.simplify_tolerance,
        parallel_threshold=args.parallel_threshold,
        neighbor_alignment=args.neighbor_alignment,

        replace_target=args.replace_target,
    )


if __name__ == "__main__":
    main()