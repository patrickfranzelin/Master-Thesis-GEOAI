from sqlalchemy import create_engine, text
import os
import json

PG_CONN = os.environ["PG_CONN"]

engine = create_engine(PG_CONN)

def write_mlqa(result: dict):

    sql = text("""
    INSERT INTO src.building_mlqa (
        building_id,
        patch_path,
        house_present,
        error_description,
        inside_pts,
        outside_pts
    )
    VALUES (
        :building_id,
        :patch_path,
        :house_present,
        :error_description,
        :inside_pts,
        :outside_pts
    )
    ON CONFLICT (building_id) DO UPDATE SET
        patch_path = EXCLUDED.patch_path,
        house_present = EXCLUDED.house_present,
        error_description = EXCLUDED.error_description,
        inside_pts = EXCLUDED.inside_pts,
        outside_pts = EXCLUDED.outside_pts,
        analyzed_at = now();
    """)


    with engine.begin() as conn:
        conn.execute(sql, {
            "building_id": result["building_id"],
            "patch_path": result["patch_path"],
            "house_present": result["house_present"],
            "error_description": result["error_description"],
            "inside_pts": json.dumps(result["inside_pts"]),
            "outside_pts": json.dumps(result["outside_pts"]),
        })
