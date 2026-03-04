"""Post-process detected building footprints with tree occlusion removal + geometric regularization.

Usage example:
    python -m src.postprocess.occlusion_regularize \
        --run-id 3a6d07e7-9307-42d0-8d2b-a53e80... \
        --source-table src.detected_house \
        --tree-table src.detected_tree \
        --target-table src.detected_house_regularized

Environment variables:
    PG_CONN: Source PostgreSQL/PostGIS connection string (required)
    OUTPUT_PG_CONN: Output PostgreSQL/PostGIS connection string (optional; defaults to PG_CONN)
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from sqlalchemy import create_engine, text

try:
    from buildingregulariser import regularize_geodataframe
except ImportError as exc:  # pragma: no cover - runtime guard for missing optional dependency
    raise ImportError(
        "buildingregulariser is not installed. Install with: pip install buildingregulariser"
    ) from exc


def remove_tree_occlusion(
    roof_poly: Polygon,
    tree_polys: Iterable[Polygon],
    buffer_size: float = 0.00001,
) -> tuple[Polygon, float]:
    """Subtract tree polygons from a roof polygon and return visible footprint + occlusion ratio."""

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
) -> int:
    """Run full post-processing for one run_id and write result table."""

    source_engine = create_engine(source_conn)

    sql = text(
        f"""
        SELECT h.id,
               h.building_id,
               h.run_id,
               h.detection_type,
               h.created_at,
               h.geom AS roof_geom,
               t.geom AS tree_geom
        FROM {source_table} h
        LEFT JOIN {tree_table} t
               ON h.building_id = t.building_id
              AND h.run_id = t.run_id
        WHERE h.run_id = :run_id
        """
    )

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
                "id": row["id"],
                "building_id": row["building_id"],
                "run_id": row["run_id"],
                "detection_type": row["detection_type"],
                "created_at": row["created_at"],
                "roof": row["roof_geom"],
                "trees": [],
            }

        if row["tree_geom"] is not None:
            grouped[key]["trees"].append(row["tree_geom"])

    base_records = []
    for entry in grouped.values():
        cleaned_geom, occlusion_ratio = remove_tree_occlusion(
            roof_poly=entry["roof"],
            tree_polys=entry["trees"],
            buffer_size=buffer_size,
        )

        base_records.append(
            {
                "source_id": entry["id"],
                "building_id": entry["building_id"],
                "run_id": entry["run_id"],
                "detection_type": entry["detection_type"],
                "source_created_at": entry["created_at"],
                "occlusion_ratio": occlusion_ratio,
                "geom": cleaned_geom,
            }
        )

    gdf = gpd.GeoDataFrame(base_records, geometry="geom", crs="EPSG:4326")

    regularized = regularize_geodataframe(
        gdf,
        parallel_threshold=parallel_threshold,
        simplify=True,
        simplify_tolerance=simplify_tolerance,
        allow_45_degree=include_45_degree,
        allow_circles=include_circles,
        neighbor_alignment=neighbor_alignment,
        neighbor_search_distance=neighbor_search_distance,
        neighbor_max_rotation=neighbor_max_rotation,
        include_metadata=True,
    )

    regularized = regularized.rename(columns={"geom": "geometry"}).set_geometry("geometry")
    regularized["area"] = regularized.geometry.area

    output_engine = create_engine(output_conn)

    if_exists = "replace" if replace_target else "append"
    regularized.to_postgis(
        name=target_table,
        con=output_engine,
        schema=target_schema,
        if_exists=if_exists,
        index=False,
    )

    print(
        f"Wrote {len(regularized)} rows to {target_schema}.{target_table} "
        f"(if_exists={if_exists})"
    )
    return len(regularized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove tree occlusion, regularize footprints, and write to a target PostGIS table."
    )
    parser.add_argument("--run-id", required=True, help="run_id in source detection tables")
    parser.add_argument(
        "--source-table",
        default="src.detected_house",
        help="Source building table (schema.table)",
    )
    parser.add_argument(
        "--tree-table",
        default="src.detected_tree",
        help="Tree detection table (schema.table)",
    )
    parser.add_argument(
        "--target-schema",
        default="src",
        help="Target schema for output table",
    )
    parser.add_argument(
        "--target-table",
        default="detected_house_regularized",
        help="Target table name for processed footprints",
    )
    parser.add_argument(
        "--replace-target",
        action="store_true",
        help="Replace target table instead of appending",
    )
    parser.add_argument(
        "--buffer-size",
        type=float,
        default=0.00001,
        help="Buffer size in degree units for morphological closing",
    )
    parser.add_argument("--parallel-threshold", type=float, default=1.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.5)
    parser.add_argument("--allow-45-degree", action="store_true")
    parser.add_argument("--allow-circles", action="store_true")
    parser.add_argument("--neighbor-alignment", action="store_true")
    parser.add_argument("--neighbor-search-distance", type=float, default=350.0)
    parser.add_argument("--neighbor-max-rotation", type=float, default=10.0)
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
    )


if __name__ == "__main__":
    main()
