from sqlalchemy import create_engine, text
from shapely.wkt import loads
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

import os

PG_CONN = os.environ["PG_CONN"]
engine = create_engine(PG_CONN)


def remove_tree_occlusion(roof_poly, tree_polys, buffer_size=0.00001):
    """
    Remove tree occlusion and regularize geometry.
    Buffer size in degrees (~1m ≈ 0.00001)
    """

    if not tree_polys:
        return roof_poly, 0.0

    tree_union = unary_union(tree_polys)

    occluded_area = roof_poly.intersection(tree_union).area
    occlusion_ratio = occluded_area / roof_poly.area

    roof_visible = roof_poly.difference(tree_union)

    if roof_visible.is_empty:
        return roof_poly, 1.0

    # keep largest fragment
    if isinstance(roof_visible, MultiPolygon):
        roof_visible = max(roof_visible.geoms, key=lambda g: g.area)

    # morphological closing
    regularized = roof_visible.buffer(buffer_size).buffer(-buffer_size)

    return regularized, occlusion_ratio


def process_buildings(run_id):

    select_sql = text("""
        SELECT h.building_id,
               ST_AsText(h.geom) AS roof_wkt,
               ST_AsText(t.geom) AS tree_wkt
        FROM src.detected_house h
        LEFT JOIN src.detected_tree t
        ON h.building_id = t.building_id
        WHERE h.run_id = :run_id
    """)

    insert_sql = text("""
        INSERT INTO src.detected_house_rectangularized
        (building_id, run_id, area, occlusion_ratio, geom)
        VALUES (
            :building_id,
            :run_id,
            :area,
            :occlusion_ratio,
            ST_SetSRID(ST_GeomFromText(:wkt), 4326)
        )
    """)

    with engine.begin() as conn:

        rows = conn.execute(select_sql, {"run_id": run_id}).fetchall()

        grouped = {}

        # Group trees per building
        for row in rows:
            bid = row.building_id
            roof = loads(row.roof_wkt)
            tree = loads(row.tree_wkt) if row.tree_wkt else None

            if bid not in grouped:
                grouped[bid] = {"roof": roof, "trees": []}

            if tree:
                grouped[bid]["trees"].append(tree)

        # Process each building
        for bid, data in grouped.items():

            roof_poly = data["roof"]
            tree_polys = data["trees"]

            final_poly, occl_ratio = remove_tree_occlusion(
                roof_poly,
                tree_polys,
                buffer_size=0.00001
            )

            conn.execute(insert_sql, {
                "building_id": bid,
                "run_id": run_id,
                "area": final_poly.area,
                "occlusion_ratio": occl_ratio,
                "wkt": final_poly.wkt,
            })

            print(f"✓ Processed building {bid} | Occlusion: {occl_ratio:.2f}")


if __name__ == "__main__":
    RUN_ID = "YOUR_RUN_ID"
    process_buildings(RUN_ID)