import json
from pathlib import Path
import shutil

import fiona
from fiona.crs import CRS
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping
from sqlalchemy import text


EPSG_4326 = CRS.from_epsg(4326)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ORIGINAL_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "building_id":        "int",
        "area_m2":            "float",
        "confidence":         "float",
        "plus_code":          "str",
        "tiff_path":          "str",
        "house_present":      "str",
        "full_house_present": "str",
        "error_description":  "str",
        "errors": "str",
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
        "errors":             "str",
        "patch_path":         "str",
        "analyzed_at":        "str",
    },
}
GLOBAL_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "detect_id":      "int",
        "building_id":    "int",
        "detection_type": "str",
        "sam_area":       "float",
        "tiff_path":      "str",
    },
}
def _check_filegdb_driver():
    supported = fiona.supported_drivers

    for driver_name in ("OpenFileGDB", "FileGDB"):
        mode = supported.get(driver_name, "")
        if "w" in mode or "W" in mode:
            return driver_name

    raise RuntimeError(
        "Writable FileGDB driver not available.\n"
        "Requires GDAL >= 3.6 (OpenFileGDB write support)."
    )

def _load_original_buildings(engine, aoi_id=None):

    where_clause = ""
    if aoi_id is not None:
        where_clause = f"""
            WHERE ST_Intersects(
                b.geom,
                (SELECT geom FROM src.aoi WHERE aoi_id = {aoi_id})
            )
        """

    sql = text(f"""
        SELECT
            b.id AS building_id,
            b.area_m2,
            b.confidence,
            b.plus_code,
            b.tiff_path,
            ST_AsText(ST_Transform(b.geom, 4326)) AS geom_wkt,
            m.house_present,
            m.full_house_present,
            m.error_description,
            m.errors,
            m.patch_path,
            m.analyzed_at
        FROM src.buildings b
        LEFT JOIN src.building_mlqa m
               ON m.building_id = b.id
        {where_clause}
        ORDER BY b.id
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    return [dict(r) for r in rows]

def _build_global_properties(row):

    def s(v): return "" if v is None else str(v)
    def f(v): return None if v is None else float(v)
    def i(v): return None if v is None else int(v)

    return {
        "detect_id":      i(row.get("detect_id")),
        "building_id":    i(row.get("building_id")),
        "detection_type": s(row.get("detection_type")),
        "sam_area":       f(row.get("sam_area")),
        "tiff_path":      s(row.get("tiff_path")),
    }

def _load_global_buildings(engine, run_id=None):

    conditions = ["d.detection_type = 'global_discovery'"]

    if run_id is not None:
        conditions.append("d.run_id = :run_id")

    where_sql = "WHERE " + " AND ".join(conditions)

    sql = text(f"""
        SELECT
            d.id AS detect_id,
            d.building_id,
            d.detection_type,
            d.area AS sam_area,
            ST_AsText(ST_Transform(d.geom, 4326)) AS geom_wkt,
            NULL AS tiff_path
        FROM src.detected_house d
        {where_sql}
        ORDER BY d.id
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"run_id": run_id}).mappings().all()

    return [dict(r) for r in rows]

def _load_improved_buildings(engine, aoi_id=None, run_id=None):

    where_clauses = []

    if aoi_id is not None:
        where_clauses.append(f"""
            ST_Intersects(
                b.geom,
                (SELECT geom FROM src.aoi WHERE aoi_id = {aoi_id})
            )
        """)

    if run_id is not None:
        where_clauses.append("d.run_id = :run_id")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = text(f"""
        SELECT
            d.id AS detect_id,
            d.building_id,
            d.detection_type,
            d.area AS sam_area,
            ST_AsText(ST_Transform(d.geom, 4326)) AS geom_wkt,
            b.area_m2,
            b.confidence,
            b.plus_code,
            b.tiff_path,
            m.house_present,
            m.full_house_present,
            m.error_description,
            m.errors,
            m.patch_path,
            m.analyzed_at
        FROM src.detected_house d
        JOIN src.buildings b
          ON b.id = d.building_id
        LEFT JOIN src.building_mlqa m
          ON m.building_id = d.building_id
        {where_sql}
        ORDER BY d.id
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"run_id": run_id}).mappings().all()

    return [dict(r) for r in rows]

def _write_layer(gdb_path, driver, layer_name, schema, rows, is_improved, is_global=False):
    written = 0
    skipped = 0

    with fiona.open(
        gdb_path,
        mode="w",
        driver=driver,
        schema=schema,
        crs=EPSG_4326,
        layer=layer_name,
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

                    if is_global:
                        properties = _build_global_properties(row)
                    else:
                        properties = _build_properties(row, is_improved)

                    dst.write({
                        "geometry": mapping(part),
                        "properties": properties,
                    })

                    written += 1

            except Exception as e:
                print(f"[WARN] Skipping feature: {e}")
                skipped += 1

    print(f"  {layer_name}: {written} written, {skipped} skipped")

def s_json(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v)

def _build_properties(row, is_improved):

    def s(v): return "" if v is None else str(v)
    def f(v): return None if v is None else float(v)
    def i(v): return None if v is None else int(v)

    props = {
        "building_id":        i(row.get("building_id")),
        "area_m2":            f(row.get("area_m2")),
        "confidence":         f(row.get("confidence")),
        "plus_code":          s(row.get("plus_code")),
        "tiff_path":          s(row.get("tiff_path")),
        "house_present":      s(row.get("house_present")),
        "full_house_present": s(row.get("full_house_present")),
        "error_description":  s(row.get("error_description")),
        "errors":             s_json(row.get("errors")),
        "patch_path":         s(row.get("patch_path")),
        "analyzed_at":        s(row.get("analyzed_at")),
    }

    if is_improved:
        props.update({
            "detect_id":      i(row.get("detect_id")),
            "detection_type": s(row.get("detection_type")),
            "sam_area":       f(row.get("sam_area")),
        })

    return props
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_buildings_to_filegdb(
    engine,
    output_path: str,
    aoi_id: int | None = None,
    run_id: str | None = None,
    overwrite: bool = True,
):
    """
    Export original and SAM-improved buildings into a FileGDB.

    Parameters
    ----------
    engine : SQLAlchemy engine
    output_path : str
        Path to output .gdb
    aoi_id : optional int
        If provided, export only buildings intersecting this AOI
    overwrite : bool
        If True, existing GDB will be deleted
    """

    driver = _check_filegdb_driver()
    gdb_path = Path(output_path).resolve()

    if gdb_path.exists() and overwrite:
        shutil.rmtree(gdb_path)

    print(f"\nExporting FileGDB → {gdb_path}")

    original_rows = _load_original_buildings(engine, aoi_id)
    improved_rows = _load_improved_buildings(engine, aoi_id, run_id)
    global_rows = _load_global_buildings(engine, run_id)

    _write_layer(
        gdb_path,
        driver,
        "original_buildings",
        ORIGINAL_SCHEMA,
        original_rows,
        is_improved=False,
    )

    _write_layer(
        gdb_path,
        driver,
        "improved_buildings",
        IMPROVED_SCHEMA,
        improved_rows,
        is_improved=True,
    )
    _write_layer(
        gdb_path,
        driver,
        "global_buildings",
        GLOBAL_SCHEMA,
        global_rows,
        is_improved=False,
        is_global=True,  # ← THIS is the key
    )

    print("✓ FileGDB export complete.")