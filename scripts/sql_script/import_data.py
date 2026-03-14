import geopandas as gpd
import rasterio
from shapely.geometry import box
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_DIR = r"C:/git/Master-Thesis-GEOAI/data/Mexico"
load_dotenv()

DB_CONN = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)
AOI_PATH = os.path.join(DATA_DIR, "aoi_mexico.gdb")
AOI_LAYER = "aoi_mexico"

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
tif_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".tif")]

BUILDINGS_CSV = os.path.join(DATA_DIR, csv_files[0])
TIFF_FILES = [os.path.join(DATA_DIR, f) for f in tif_files]

AOI_EXISTS = os.path.exists(AOI_PATH)

print("Detected CSV:", BUILDINGS_CSV)
print("Detected TIFFs:", len(TIFF_FILES))

# --------------------------------------------------
# CONNECT DB
# --------------------------------------------------

engine = create_engine(DB_CONN)
conn = engine.raw_connection()
cursor = conn.cursor()

# --------------------------------------------------
# CREATE EXTENSIONS + SCHEMA
# --------------------------------------------------

cursor.execute("""
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS src;
""")

conn.commit()

cursor.execute("""
-- --------------------------------------------------
-- BUILDINGS TABLE
-- --------------------------------------------------

CREATE TABLE IF NOT EXISTS src.buildings(
    id SERIAL PRIMARY KEY,
    geom geometry(MULTIPOLYGON,4326),
    area_m2 DOUBLE PRECISION,
    confidence REAL,
    plus_code TEXT,
    tiff_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_buildings_geom
ON src.buildings USING GIST(geom);
""")

conn.commit()

# --------------------------------------------------
# CREATE SCHEMA
# --------------------------------------------------

cursor.execute("""
-- --------------------------------------------------
-- DETECTED HOUSE TABLE
-- --------------------------------------------------

CREATE TABLE IF NOT EXISTS src.detected_house(
    id SERIAL PRIMARY KEY,
    building_id INTEGER,
    detection_type TEXT,
    area FLOAT8,
    geom geometry(POLYGON,4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_detected_house_geom
ON src.detected_house USING GIST(geom);
    
-- --------------------------------------------------
-- DETECTED HOUSE REGULARIZED
-- --------------------------------------------------

CREATE TABLE IF NOT EXISTS src.detected_house_regularized(
    id SERIAL PRIMARY KEY,
    building_id INTEGER,
    area FLOAT8,
    geom geometry(POLYGON,4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_detected_house_regularized_geom
ON src.detected_house_regularized USING GIST(geom);

-- --------------------------------------------------
-- DETECTED TREE TABLE
-- --------------------------------------------------

CREATE TABLE IF NOT EXISTS src.detected_tree(
    id SERIAL PRIMARY KEY,
    building_id INTEGER,
    area FLOAT8,
    geom geometry(POLYGON,4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_detected_tree_geom
ON src.detected_tree USING GIST(geom);
""")

conn.commit()

# --------------------------------------------------
# LOAD AOI
# --------------------------------------------------

if AOI_EXISTS:

    print("Loading AOI...")

    aoi = gpd.read_file(AOI_PATH, layer=AOI_LAYER).to_crs(4326)
    aoi = aoi.rename(columns={"geometry": "geom"}).set_geometry("geom")

    aoi.to_postgis("aoi", engine, schema="src", if_exists="replace", index=False)

    minx, miny, maxx, maxy = aoi.total_bounds
    print("AOI bbox:", minx, miny, maxx, maxy)

else:

    print("No AOI found")

# --------------------------------------------------
# REGISTER TIFFS
# --------------------------------------------------

print("Registering TIFF footprints...")

records = []

for path in TIFF_FILES:

    with rasterio.open(path) as src:

        bounds = src.bounds

        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        records.append({
            "path": path,
            "geometry": geom
        })

tiffs = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

tiffs.to_postgis("tiffs", engine, schema="src", if_exists="replace", index=False)

cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiffs_geom ON src.tiffs USING GIST(geometry);")

if AOI_EXISTS:
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_aoi_geom ON src.aoi USING GIST(geom);")

conn.commit()

print("TIFFs registered:", len(records))

# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

print("Reading CSV...")

df = pd.read_csv(BUILDINGS_CSV)

print("Total rows:", len(df))

# --------------------------------------------------
# AOI PRE-FILTER
# --------------------------------------------------

if AOI_EXISTS:

    print("Pre-filtering by AOI bounding box...")

    coords = df["geometry"].str.extract(
        r'POLYGON\(\((-?\d+\.?\d*) (-?\d+\.?\d*)'
    ).astype(float)

    df["lon"] = coords[0]
    df["lat"] = coords[1]

    before = len(df)

    df = df[
        (df["lon"] >= minx) &
        (df["lon"] <= maxx) &
        (df["lat"] >= miny) &
        (df["lat"] <= maxy)
    ]

    print("After bbox filter:", len(df), "/", before)

# --------------------------------------------------
# CONVERT GEOMETRY
# --------------------------------------------------

print("Converting WKT → geometry")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
    crs="EPSG:4326"
)

gdf = gdf.rename(columns={"geometry": "geom"}).set_geometry("geom")

gdf = gdf[[
    "geom",
    "area_in_meters",
    "confidence",
    "full_plus_code"
]]

gdf.rename(columns={
    "area_in_meters": "area_m2",
    "full_plus_code": "plus_code"
}, inplace=True)

print("Buildings prepared:", len(gdf))

# --------------------------------------------------
# UPLOAD STAGING
# --------------------------------------------------

print("Uploading staging table...")

gdf.to_postgis(
    "_staging_buildings",
    engine,
    schema="src",
    if_exists="replace",
    index=False
)

print("Staging uploaded")

# --------------------------------------------------
# INSERT BUILDINGS + TIFF LINK
# --------------------------------------------------

print("Inserting buildings with TIFF match...")

cursor.execute("""
INSERT INTO src.buildings (geom, area_m2, confidence, plus_code, tiff_path)

SELECT
    ST_Multi(s.geom),
    s.area_m2,
    s.confidence,
    s.plus_code,
    t.path

FROM src._staging_buildings s
CROSS JOIN src.tiffs t;
""")

conn.commit()

# --------------------------------------------------
# INDEX
# --------------------------------------------------

cursor.execute("CREATE INDEX IF NOT EXISTS idx_buildings_geom ON src.buildings USING GIST(geom);")
conn.commit()

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cursor.execute("DROP TABLE src._staging_buildings;")
conn.commit()

cursor.execute("SELECT COUNT(*) FROM src.buildings")
count = cursor.fetchone()[0]

print("Buildings stored:", count)

print("Import finished.")