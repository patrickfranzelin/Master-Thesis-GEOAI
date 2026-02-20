"""
export_to_filegdb.py
--------------------
Exports building polygons from PostgreSQL into an Esri File Geodatabase (FileGDB)
with two feature classes:

  1. original_buildings  – every polygon from src.buildings joined with its
                           MLQA result (house_present, full_house_present, …).
                           Includes buildings marked as deleted / not-a-house
                           (house_present = False or NULL).

  2. improved_buildings  – SAM-refined polygons from src.detected_house joined
                           with the corresponding MLQA result and building
                           attributes.

Requirements
------------
- Environment variable PG_CONN must point to a valid PostgreSQL connection string,
  e.g. "postgresql://user:pass@host:5432/dbname".
- GDAL / Fiona must be built with the OpenFileGDB (write) driver.
  This is available in GDAL >= 3.6 / Fiona >= 1.9.

Usage
-----
    python scripts/export_to_filegdb.py [--output path/to/output.gdb] [--aoi-id 3]
"""

import argparse
import os
import sys
from pathlib import Path

import fiona
import fiona.crs
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPSG_4326 = fiona.crs.from_epsg(4326)

ORIGINAL_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "building_id":        "int",
        "area_m2":            "float",
        "confidence":         "float",
        "plus_code":          "str",
        "tiff_path":          "str",
        "house_present":      "str",   # stored as string; True/False/NULL
        "full_house_present": "str",
        "error_description":  "str",
        "patch_path":         "str",
        "analyzed_at":        "str",
    },
}

IMPROVED_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "detect_id":          "int",
        "building_id":        "int",
        "detection_type":     "str",
        "sam_area":           "float",
        "area_m2":            "float",
        "confidence":         "float",
        "plus_code":          "str",
        "tiff_path":          "str",
        "house_present":      "str",
        "full_house_present": "str",
        "error_description":  "str",
        "patch_path":         "str",
        "analyzed_at":        "str",
    },
}


def _bool_str(value) -> str:
    """Convert a Python bool / None to a readable string for FileGDB."""
    if value is None:
        return "NULL"
    return str(value)


def _str_or_none(value) -> str:
    if value is None:
        return ""
    return str(value)


def _float_or_none(value):
    if value is None:
        return None
    return float(value)


def _int_or_none(value):
    if value is None:
        return None
    return int(value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_original_buildings(engine) -> list[dict]:
    """
    Return all buildings from src.buildings LEFT-JOINed with src.building_mlqa.
    Includes buildings that have no MLQA entry (not yet analysed) and those
    flagged as not a house (house_present = FALSE).
    """
    sql = text("""
        SELECT
            b.id                  AS building_id,
            b.area_m2,
            b.confidence,
            b.plus_code,
            b.tiff_path,
            ST_AsText(b.geom)     AS geom_wkt,
            m.house_present,
            m.full_house_present,
            m.error_description,
            m.patch_path,
            m.analyzed_at
        FROM src.buildings b
        LEFT JOIN src.building_mlqa m
               ON m.building_id = b.id
        ORDER BY b.id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [dict(r) for r in rows]


def load_improved_buildings(engine) -> list[dict]:
    """
    Return all SAM-detected polygons from src.detected_house joined with
    src.buildings and src.building_mlqa attributes.
    """
    sql = text("""
        SELECT
            d.id                  AS detect_id,
            d.building_id,
            d.detection_type,
            d.area                AS sam_area,
            ST_AsText(d.geom)     AS geom_wkt,
            b.area_m2,
            b.confidence,
            b.plus_code,
            b.tiff_path,
            m.house_present,
            m.full_house_present,
            m.error_description,
            m.patch_path,
            m.analyzed_at
        FROM src.detected_house d
        JOIN src.buildings b
          ON b.id = d.building_id
        LEFT JOIN src.building_mlqa m
               ON m.building_id = d.building_id
        ORDER BY d.id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# FileGDB writing
# ---------------------------------------------------------------------------

def _check_filegdb_driver():
    """Verify that Fiona can write OpenFileGDB; raise if not available."""
    supported = fiona.supported_drivers
    write_driver = None
    for driver_name in ("OpenFileGDB", "FileGDB"):
        mode = supported.get(driver_name, "")
        if "w" in mode or "W" in mode:
            write_driver = driver_name
            break
    if write_driver is None:
        raise RuntimeError(
            "No writable FileGDB driver found in Fiona/GDAL.\n"
            "Requires GDAL >= 3.6 (OpenFileGDB write support) or the "
            "proprietary FileGDB driver.\n"
            f"Available drivers: {list(supported.keys())}"
        )
    return write_driver


def write_original_buildings(gdb_path: str, driver: str, rows: list[dict]):
    """Write original_buildings feature class into the FileGDB."""
    layer = "original_buildings"
    written = 0
    skipped = 0

    with fiona.open(
        gdb_path,
        mode="w",
        driver=driver,
        schema=ORIGINAL_SCHEMA,
        crs=EPSG_4326,
        layer=layer,
    ) as dst:
        for row in rows:
            wkt = row.get("geom_wkt")
            if not wkt:
                skipped += 1
                continue
            try:
                geom = shapely_wkt.loads(wkt)
                if geom.geom_type == "MultiPolygon":
                    # Flatten multipolygons: write each part as a separate feature
                    parts = list(geom.geoms)
                elif geom.geom_type == "Polygon":
                    parts = [geom]
                else:
                    skipped += 1
                    continue
                for part in parts:
                    dst.write({
                        "geometry": mapping(part),
                        "properties": {
                            "building_id":        _int_or_none(row["building_id"]),
                            "area_m2":            _float_or_none(row.get("area_m2")),
                            "confidence":         _float_or_none(row.get("confidence")),
                            "plus_code":          _str_or_none(row.get("plus_code")),
                            "tiff_path":          _str_or_none(row.get("tiff_path")),
                            "house_present":      _bool_str(row.get("house_present")),
                            "full_house_present": _bool_str(row.get("full_house_present")),
                            "error_description":  _str_or_none(row.get("error_description")),
                            "patch_path":         _str_or_none(row.get("patch_path")),
                            "analyzed_at":        _str_or_none(row.get("analyzed_at")),
                        },
                    })
                    written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Skipping building_id={row.get('building_id')}: {exc}")
                skipped += 1

    print(f"  original_buildings: {written} features written, {skipped} skipped")


def write_improved_buildings(gdb_path: str, driver: str, rows: list[dict]):
    """Write improved_buildings feature class into the FileGDB."""
    layer = "improved_buildings"
    written = 0
    skipped = 0

    with fiona.open(
        gdb_path,
        mode="w",
        driver=driver,
        schema=IMPROVED_SCHEMA,
        crs=EPSG_4326,
        layer=layer,
    ) as dst:
        for row in rows:
            wkt = row.get("geom_wkt")
            if not wkt:
                skipped += 1
                continue
            try:
                geom = shapely_wkt.loads(wkt)
                if geom.geom_type == "MultiPolygon":
                    parts = list(geom.geoms)
                elif geom.geom_type == "Polygon":
                    parts = [geom]
                else:
                    skipped += 1
                    continue
                for part in parts:
                    dst.write({
                        "geometry": mapping(part),
                        "properties": {
                            "detect_id":          _int_or_none(row.get("detect_id")),
                            "building_id":        _int_or_none(row["building_id"]),
                            "detection_type":     _str_or_none(row.get("detection_type")),
                            "sam_area":           _float_or_none(row.get("sam_area")),
                            "area_m2":            _float_or_none(row.get("area_m2")),
                            "confidence":         _float_or_none(row.get("confidence")),
                            "plus_code":          _str_or_none(row.get("plus_code")),
                            "tiff_path":          _str_or_none(row.get("tiff_path")),
                            "house_present":      _bool_str(row.get("house_present")),
                            "full_house_present": _bool_str(row.get("full_house_present")),
                            "error_description":  _str_or_none(row.get("error_description")),
                            "patch_path":         _str_or_none(row.get("patch_path")),
                            "analyzed_at":        _str_or_none(row.get("analyzed_at")),
                        },
                    })
                    written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Skipping detect_id={row.get('detect_id')}: {exc}")
                skipped += 1

    print(f"  improved_buildings: {written} features written, {skipped} skipped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export building polygons from PostgreSQL to an Esri File Geodatabase."
    )
    parser.add_argument(
        "--output",
        default="buildings_export.gdb",
        help="Path to the output File Geodatabase (default: buildings_export.gdb)",
    )
    parser.add_argument(
        "--pg-conn",
        default=None,
        help=(
            "PostgreSQL connection string. "
            "If not provided, the PG_CONN environment variable is used."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    pg_conn = args.pg_conn or os.environ.get("PG_CONN")
    if not pg_conn:
        print(
            "ERROR: No PostgreSQL connection string provided.\n"
            "Set the PG_CONN environment variable or pass --pg-conn.",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = create_engine(pg_conn)

    # ------------------------------------------------------------------
    # Driver check
    # ------------------------------------------------------------------
    print("Checking FileGDB write driver …")
    driver = _check_filegdb_driver()
    print(f"  Using driver: {driver}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading original buildings from PostgreSQL …")
    original_rows = load_original_buildings(engine)
    print(f"  {len(original_rows)} rows loaded")

    print("Loading improved (SAM-detected) buildings from PostgreSQL …")
    improved_rows = load_improved_buildings(engine)
    print(f"  {len(improved_rows)} rows loaded")

    # ------------------------------------------------------------------
    # Write FileGDB
    # ------------------------------------------------------------------
    gdb_path = str(Path(args.output).resolve())
    print(f"Writing FileGDB: {gdb_path}")

    write_original_buildings(gdb_path, driver, original_rows)
    write_improved_buildings(gdb_path, driver, improved_rows)

    print("\nDone. FileGDB written to:", gdb_path)
    print("Feature classes:")
    print("  • original_buildings  – all original polygons (incl. deleted / not-a-house)")
    print("  • improved_buildings  – SAM-refined polygons")


if __name__ == "__main__":
    main()
