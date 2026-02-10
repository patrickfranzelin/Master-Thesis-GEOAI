from sqlalchemy import create_engine, text
import os
import json

PG_CONN = os.environ["PG_CONN"]

engine = create_engine(PG_CONN)

def write_mlqa(result: dict):
    """
    Write MLQA analysis results to database.
    
    Args:
        result: Dictionary containing:
            - building_id: int
            - patch_path: str (optional, can be None)
            - house_present: bool
            - full_house_present: bool or None
            - error_description: str or None
            - inside_pts: list
            - outside_pts: list
    """

    sql = text("""
    INSERT INTO src.building_mlqa (
        building_id,
        patch_path,
        house_present,
        full_house_present,
        error_description,
        inside_pts,
        outside_pts
    )
    VALUES (
        :building_id,
        :patch_path,
        :house_present,
        :full_house_present,
        :error_description,
        :inside_pts,
        :outside_pts
    )
    ON CONFLICT (building_id) DO UPDATE SET
        patch_path = EXCLUDED.patch_path,
        house_present = EXCLUDED.house_present,
        full_house_present = EXCLUDED.full_house_present,
        error_description = EXCLUDED.error_description,
        inside_pts = EXCLUDED.inside_pts,
        outside_pts = EXCLUDED.outside_pts,
        analyzed_at = now();
    """)


    with engine.begin() as conn:
        conn.execute(sql, {
            "building_id": result["building_id"],
            "patch_path": result.get("patch_path"),
            "house_present": result["house_present"],
            "full_house_present": result.get("full_house_present"),
            "error_description": result.get("error_description"),
            "inside_pts": json.dumps(result.get("inside_pts", [])),
            "outside_pts": json.dumps(result.get("outside_pts", [])),
        })
