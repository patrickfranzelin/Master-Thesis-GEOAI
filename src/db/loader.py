import geopandas as gpd
from sqlalchemy import create_engine
import os

PG_CONN = os.environ["PG_CONN"]

def load_buildings(limit=None):
    engine = create_engine(PG_CONN)

    sql = """
    SELECT id, geom, tiff_path
    FROM src.buildings
    WHERE tiff_path IS NOT NULL
    """

    if limit:
        sql += f" LIMIT {limit}"

    gdf = gpd.read_postgis(sql, engine, geom_col="geom")

    print(f"Loaded {len(gdf)} buildings with TIFF")
    print("Buildings CRS:", gdf.crs)

    return gdf
