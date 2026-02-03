from sqlalchemy import create_engine, text

PG_CONN = "postgresql://postgres:2712@localhost:5432/geoai"

engine = create_engine(PG_CONN)


def write_mlqa(result: dict):

    sql = text("""
    INSERT INTO src.building_mlqa (
        building_id,
        error_description
    )
    VALUES (
        :building_id,
        :error_description
    )
    ON CONFLICT (building_id) DO UPDATE SET
        error_description = EXCLUDED.error_description,
        analyzed_at = now();
    """)

    with engine.begin() as conn:
        conn.execute(sql, {
            "building_id": result["building_id"],
            "error_description": result["error_description"]
        })

