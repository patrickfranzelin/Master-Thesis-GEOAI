from pathlib import Path
import json
import cv2
import numpy as np
import geopandas as gpd
from ultralytics import SAM
from shapely.geometry import Polygon
from buildingregulariser import regularize_geodataframe

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(r"C:\git\Master-Thesis-GEOAI")

IMAGE = PROJECT_ROOT / "data/img.png"
POINTS_JSON = PROJECT_ROOT / "outputs/db_results/points/test.json"
SAM_MODEL = PROJECT_ROOT / "models/sam3_weights/sam3.pt"

MASK_OUT = PROJECT_ROOT / "outputs/sam_test_mask.png"
OVERLAY_OUT = PROJECT_ROOT / "outputs/sam_polygon_overlay.png"

# --------------------------------------------------
# Load prompt points
# --------------------------------------------------

with open(POINTS_JSON) as f:
    pts = json.load(f)["inside"]

labels = [1] * len(pts)

# --------------------------------------------------
# Run SAM
# --------------------------------------------------

print("Loading SAM...")
model = SAM(str(SAM_MODEL))

print("Running SAM...")
results = model.predict(
    source=str(IMAGE),
    points=pts,
    labels=labels,
)

if results[0].masks is None:
    raise RuntimeError("SAM produced no mask")

mask = results[0].masks.data[0].cpu().numpy()
mask = (mask * 255).astype(np.uint8)

cv2.imwrite(str(MASK_OUT), mask)
print("Saved raw mask")

# --------------------------------------------------
# Morphological cleanup (vegetation / dents)
# --------------------------------------------------

kernel = np.ones((7,7), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# --------------------------------------------------
# Largest contour → shapely polygon
# --------------------------------------------------

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

raw_poly = Polygon(cnt.squeeze())

# remove pixel staircases before GIS regularisation
raw_poly = raw_poly.simplify(3.0, preserve_topology=True)

# --------------------------------------------------
# GeoDataFrame (pixel CRS placeholder)
# --------------------------------------------------

gdf = gpd.GeoDataFrame(geometry=[raw_poly], crs="EPSG:3857")

# --------------------------------------------------
# REGULARISE BUILDING
# --------------------------------------------------

reg = regularize_geodataframe(
    gdf,
    simplify_tolerance=8.0,     # 6–10 typical
    allow_45_degree=True,
    allow_circles=False,
    neighbor_alignment=False,
    num_cores=1               # Windows safe
)

poly = reg.geometry.iloc[0]

# --------------------------------------------------
# Draw result
# --------------------------------------------------

img = cv2.imread(str(IMAGE))

pts_poly = np.array(poly.exterior.coords).astype(np.int32)

cv2.polylines(img, [pts_poly], True, (0,255,0), 3)

cv2.imwrite(str(OVERLAY_OUT), img)

print("Saved overlay:", OVERLAY_OUT)
