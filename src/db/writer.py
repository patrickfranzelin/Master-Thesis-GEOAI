from sqlalchemy import create_engine, text
import os
import json
import rasterio
from shapely.affinity import affine_transform
from shapely.ops import transform as shp_transform
import pyproj

PG_CONN = os.environ["PG_CONN"]

engine = create_engine(PG_CONN)

def write_mlqa(result: dict):
    """
    Write MLQA analysis results to database.
    
    Args:
        result: Dictionary containing:
            - building_id: int
            - patch_path: str (optional, can be None)
            - house_present: bool or None (None indicates parse error/uncertainty)
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


def write_detected_houses(
        building_id: int,
        polygons,
        detection_type: str,
        run_id: str,
        tiff_path: str,
        win,
):
    """
    Polygons are in resized PATCH PIXEL coordinates (512x512).
    We reverse:
        1. resize
        2. window offset
        3. raster affine
        4. reprojection
    """

    if not polygons:
        return

    from shapely.affinity import translate, scale
    from shapely.ops import transform as shp_transform

    sql = text("""
        INSERT INTO src.detected_house (
            building_id,
            detection_type,
            area,
            geom,
            run_id
        )
        VALUES (
            :building_id,
            :detection_type,
            :area,
            ST_SetSRID(ST_GeomFromText(:wkt), 4326),
            :run_id
        )
    """)

    with rasterio.open(tiff_path) as src:

        transform = src.transform
        raster_crs = src.crs

        affine_params = [
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.xoff,
            transform.yoff,
        ]

        # reprojection
        if raster_crs.to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(
                raster_crs,
                "EPSG:4326",
                always_xy=True
            )
            project = transformer.transform
        else:
            project = None

        with engine.begin() as conn:
            for poly in polygons:

                if poly is None:
                    continue

                # 1️⃣ Undo resize (512 → window size)
                sx = win.width / 512
                sy = win.height / 512
                poly_unscaled = scale(poly, xfact=sx, yfact=sy, origin=(0, 0))

                # 2️⃣ Add window offset (patch → full raster pixel)
                poly_full_px = translate(
                    poly_unscaled,
                    xoff=win.col_off,
                    yoff=win.row_off
                )

                # 3️⃣ Pixel → raster CRS (UTM meters)
                utm_poly = affine_transform(poly_full_px, affine_params)

                # 4️⃣ Raster CRS → WGS84
                if project:
                    geo_poly = shp_transform(project, utm_poly)
                else:
                    geo_poly = utm_poly

                conn.execute(sql, {
                    "building_id": building_id,
                    "detection_type": detection_type,
                    "area": utm_poly.area,  # keep metric area
                    "wkt": geo_poly.wkt,
                    "run_id": run_id,
                })