import cv2
import rasterio as rio
import numpy as np
import geopandas as gpd

from shapely.ops import unary_union, transform
from rasterio.windows import from_bounds
from pyproj import Transformer


def extract_patch(geom, geom_crs, raster_path, out_size=512):

    with rio.open(raster_path) as src:

        # Geometry → raster CRS
        to_raster = Transformer.from_crs(geom_crs, src.crs, always_xy=True).transform
        geom_raster = transform(to_raster, geom)

        minx, miny, maxx, maxy = geom_raster.bounds

        # center
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        context = 2 # was 1.2

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

        return img, geom_px

