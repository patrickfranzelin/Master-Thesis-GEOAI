import cv2
import rasterio as rio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import unary_union, transform
from rasterio.windows import from_bounds
from pyproj import Transformer


def extract_patch(geom, geom_crs, raster_path, out_size=512, context=2):

    with rio.open(raster_path) as src:

        # Geometry → raster CRS
        to_raster = Transformer.from_crs(geom_crs, src.crs, always_xy=True).transform
        geom_raster = transform(to_raster, geom)

        minx, miny, maxx, maxy = geom_raster.bounds

        # center
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        # context factor increased from 2 to 3 to better capture full houses even with inaccurate polygons

        size = max(maxx - minx, maxy - miny) * context

        half = size / 2

        # square bounds
        win = from_bounds(
            cx - half,
            cy - half,
            cx + half,
            cy + half,
            src.transform,
        ).round_offsets().round_lengths()

        # read WITHOUT boundless padding
        data = src.read([1, 2, 3], window=win)
        img = np.moveaxis(data, 0, -1)

        # resize directly
        h, w = img.shape[:2]
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

        # polygon → pixel coords
        inv = ~src.window_transform(win)

        geom_px = transform(lambda x, y: inv * (x, y), geom_raster)

        sx = out_size / w
        sy = out_size / h

        geom_px = transform(lambda x, y: (x * sx, y * sy), geom_px)

        return img, geom_px, win

def extract_patch_pixel(img, poly_px, out_size=512, context=2.0):
    h_img, w_img = img.shape[:2]
    minx, miny, maxx, maxy = poly_px.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    size = max(maxx - minx, maxy - miny) * context
    half = size / 2

    x1 = int(max(cx - half, 0))
    y1 = int(max(cy - half, 0))
    x2 = int(min(cx + half, w_img))
    y2 = int(min(cy + half, h_img))

    crop = img[y1:y2, x1:x2]
    h_crop, w_crop = crop.shape[:2]
    crop_resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    from shapely.affinity import translate, scale
    poly_shifted = translate(poly_px, xoff=-x1, yoff=-y1)
    sx = out_size / w_crop
    sy = out_size / h_crop
    poly_rescaled = scale(poly_shifted, xfact=sx, yfact=sy, origin=(0, 0))

    #  return crop info so caller can invert back to img_big space
    crop_info = (x1, y1, w_crop, h_crop)
    return crop_resized, poly_rescaled, crop_info

def extract_aoi_raster(aoi_geom, aoi_crs, tiff_path):
    """
    Extract full AOI raster clipped to polygon.

    Returns:
        img (H, W, C) uint8 RGB
        transform (Affine)
        crs
    """

    with rio.open(tiff_path) as src:

        # Reproject AOI if needed
        if aoi_crs != src.crs:
            import geopandas as gpd
            aoi_geom = gpd.GeoSeries([aoi_geom], crs=aoi_crs).to_crs(src.crs).iloc[0]

        # Mask raster to AOI
        out_image, out_transform = mask(
            src,
            [mapping(aoi_geom)],
            crop=True
        )

        # Convert CHW → HWC
        img = out_image.transpose(1, 2, 0)

        # Normalize to uint8 if needed
        if img.dtype != np.uint8:
            img = img.astype("float32")
            img = img / (img.max() + 1e-6) * 255
            img = img.astype("uint8")

    return img, out_transform, src.crs