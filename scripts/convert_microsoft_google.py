import gzip
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_DIR = Path(r"C:\git\Master-Thesis-GEOAI\data\switzerland_buildings")

OUT_CSV = DATA_DIR / "switzerland_buildings.csv"
OUT_GPKG = DATA_DIR / "switzerland_buildings.gpkg"

METRIC_CRS = "EPSG:4326"  # Liberia UTM Zone 29N

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def read_microsoft_geojsonl_gz(path: Path) -> gpd.GeoDataFrame:
    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            geom = shape(obj["geometry"])
            props = obj.get("properties", {})

            rows.append({
                "confidence": props.get("confidence", -1.0),
                "geometry": geom
            })

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def to_google_wkt(geom) -> str:
    # Shapely normally writes "POLYGON ((...))"
    # Google file style is "POLYGON((...))"
    return geom.wkt.replace("POLYGON ((", "POLYGON((").replace("MULTIPOLYGON (((", "MULTIPOLYGON(((")


# --------------------------------------------------
# READ RAW MICROSOFT FILES ONLY
# --------------------------------------------------

files = sorted(DATA_DIR.glob("*.csv.gz"))

print(f"Found {len(files)} Microsoft .csv.gz files")

all_gdfs = []

for file in files:
    print("Reading:", file.name)
    gdf_part = read_microsoft_geojsonl_gz(file)

    if len(gdf_part) > 0:
        all_gdfs.append(gdf_part)
        print(f"  loaded {len(gdf_part):,} buildings")

if not all_gdfs:
    raise ValueError("No valid Microsoft .csv.gz files found.")

# --------------------------------------------------
# COMBINE
# --------------------------------------------------

gdf = gpd.GeoDataFrame(
    pd.concat(all_gdfs, ignore_index=True),
    geometry="geometry",
    crs="EPSG:4326"
)

print("Total before cleaning:", len(gdf))

gdf = gdf[gdf.geometry.notna()]
gdf = gdf[~gdf.geometry.is_empty]
gdf["geometry"] = gdf.geometry.make_valid()

# keep polygons only
gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

# remove exact duplicates
gdf["wkt_tmp"] = gdf.geometry.to_wkt()
gdf = gdf.drop_duplicates(subset="wkt_tmp").drop(columns="wkt_tmp")

print("Total after cleaning:", len(gdf))

# --------------------------------------------------
# CREATE GOOGLE-LIKE COLUMNS
# --------------------------------------------------

gdf_metric = gdf.to_crs(METRIC_CRS)

centroids_wgs84 = gpd.GeoSeries(
    gdf_metric.geometry.centroid,
    crs=METRIC_CRS
).to_crs("EPSG:4326")

out_df = pd.DataFrame({
    "latitude": centroids_wgs84.y.round(8),
    "longitude": centroids_wgs84.x.round(8),
    "area_in_meters": gdf_metric.geometry.area.round(4),
    "confidence": gdf["confidence"].astype(float).round(4),
    "geometry": gdf.geometry.apply(to_google_wkt),
    "full_plus_code": ""
})

# exact column order
out_df = out_df[[
    "latitude",
    "longitude",
    "area_in_meters",
    "confidence",
    "geometry",
    "full_plus_code"
]]

# --------------------------------------------------
# SAVE
# --------------------------------------------------

out_df.to_csv(OUT_CSV, index=False)

gdf_out = gpd.GeoDataFrame(
    out_df.drop(columns="geometry"),
    geometry=gdf.geometry,
    crs="EPSG:4326"
)

gdf_out.to_file(OUT_GPKG, driver="GPKG")

print("Saved CSV:", OUT_CSV)
print("Saved GPKG:", OUT_GPKG)
print(out_df.head())