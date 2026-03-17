from sqlalchemy import create_engine, text
import os
import json
import rasterio
from shapely.affinity import affine_transform
import pyproj
from shapely.affinity import translate, scale
from shapely.ops import transform as shp_transform

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
               INSERT INTO src.building_mlqa (building_id,
                                              patch_path,
                                              house_present,
                                              full_house_present,
                                              error_description,
                                              errors, -- NEW
                                              inside_pts,
                                              outside_pts)
               VALUES (:building_id,
                       :patch_path,
                       :house_present,
                       :full_house_present,
                       :error_description,
                       :errors, -- NEW
                       :inside_pts,
                       :outside_pts) ON CONFLICT (building_id) DO
               UPDATE SET
                   patch_path = EXCLUDED.patch_path,
                   house_present = EXCLUDED.house_present,
                   full_house_present = EXCLUDED.full_house_present,
                   error_description = EXCLUDED.error_description,
                   errors = EXCLUDED.errors, -- NEW
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
            "errors": json.dumps(result.get("errors", [])),
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
        metadata: dict = None,
):
    """
    Polygons backprojection chain:
      PARTIAL: refine_img (512) → img_big (512) → win_big px → raster CRS → WGS84
      FULL:    patch (512)                       → win px     → raster CRS → WGS84
    """

    if not polygons:
        return

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

    crop_info = metadata.get("crop_info") if metadata else None

    with rasterio.open(tiff_path) as src:

        raster_transform = src.transform
        raster_crs = src.crs

        affine_params = [
            raster_transform.a,
            raster_transform.b,
            raster_transform.d,
            raster_transform.e,
            raster_transform.xoff,
            raster_transform.yoff,
        ]

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

                # ── PARTIAL only: undo refine sub-crop ──────────────────────
                # poly lives in refine_img space (512×512 sub-crop of img_big)
                # must map back to img_big space (also 512×512) first
                if crop_info is not None:
                    x1, y1, w_crop, h_crop = crop_info

                    #  undo resize: refine 512×512 → actual crop size
                    poly = scale(
                        poly,
                        xfact=w_crop / 512,
                        yfact=h_crop / 512,
                        origin=(0, 0),
                    )

                    # undo crop offset: crop space → img_big pixel space
                    poly = translate(poly, xoff=x1, yoff=y1)

                # ── ALL pipelines: undo img→win resize ──────────────────────
                # img_big / patch img was resized from win pixel size to 512×512
                #  undo resize: 512×512 → win pixel dimensions
                sam_size = metadata.get("sam_input_size", 512)

                sx = win.width / sam_size
                sy = win.height / sam_size
                poly_unscaled = scale(poly, xfact=sx, yfact=sy, origin=(0, 0))

                #  add window offset: win-relative px → full raster pixel
                poly_full_px = translate(
                    poly_unscaled,
                    xoff=win.col_off,
                    yoff=win.row_off,
                )

                #  pixel → raster CRS (UTM / local metres)
                utm_poly = affine_transform(poly_full_px, affine_params)

                #  raster CRS → WGS84
                geo_poly = shp_transform(project, utm_poly) if project else utm_poly

                conn.execute(sql, {
                    "building_id": building_id,
                    "detection_type": detection_type,
                    "area": utm_poly.area,
                    "wkt": geo_poly.wkt,
                    "run_id": run_id,
                })

                print("WIN WIDTH:", win.width)
                print("WIN HEIGHT:", win.height)
                print("Poly bounds (pixel space):", poly.bounds)

def write_detected_trees(
        building_id: int,
        polygons,
        run_id: str,
        tiff_path: str,
        win,
        metadata: dict = None,
):
    """
    Backprojects tree polygons from SAM pixel space to WGS84.

    Uses identical projection chain as write_detected_houses.
    """

    if not polygons:
        return

    sql = text("""
        INSERT INTO src.detected_tree (
            building_id,
            area,
            geom,
            run_id
        )
        VALUES (
            :building_id,
            :area,
            ST_SetSRID(ST_GeomFromText(:wkt), 4326),
            :run_id
        )
    """)

    crop_info = metadata.get("crop_info") if metadata else None
    sam_size = metadata.get("sam_input_size", 512)

    with rasterio.open(tiff_path) as src:

        raster_transform = src.transform
        raster_crs = src.crs

        affine_params = [
            raster_transform.a,
            raster_transform.b,
            raster_transform.d,
            raster_transform.e,
            raster_transform.xoff,
            raster_transform.yoff,
        ]

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

                if poly is None or poly.is_empty:
                    continue

                # ─────────────────────────────────────────────
                # 1️⃣ Undo refine crop (PARTIAL only)
                # ─────────────────────────────────────────────
                if crop_info is not None:
                    x1, y1, w_crop, h_crop = crop_info

                    poly = scale(
                        poly,
                        xfact=w_crop / 512,
                        yfact=h_crop / 512,
                        origin=(0, 0),
                    )

                    poly = translate(poly, xoff=x1, yoff=y1)

                # ─────────────────────────────────────────────
                # 2️⃣ Undo SAM resize
                # ─────────────────────────────────────────────
                sx = win.width / sam_size
                sy = win.height / sam_size

                poly_unscaled = scale(
                    poly,
                    xfact=sx,
                    yfact=sy,
                    origin=(0, 0),
                )

                # ─────────────────────────────────────────────
                # 3️⃣ Add window offset
                # ─────────────────────────────────────────────
                poly_full_px = translate(
                    poly_unscaled,
                    xoff=win.col_off,
                    yoff=win.row_off,
                )

                # ─────────────────────────────────────────────
                # 4️⃣ Pixel → raster CRS
                # ─────────────────────────────────────────────
                utm_poly = affine_transform(poly_full_px, affine_params)

                # ─────────────────────────────────────────────
                # 5️⃣ Raster CRS → WGS84
                # ─────────────────────────────────────────────
                geo_poly = (
                    shp_transform(project, utm_poly)
                    if project else utm_poly
                )

                conn.execute(sql, {
                    "building_id": building_id,
                    "area": utm_poly.area,
                    "wkt": geo_poly.wkt,
                    "run_id": run_id,
                })