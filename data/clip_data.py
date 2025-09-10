import rasterio
from rasterio.windows import from_bounds

# Bounding Box (Beispiel: LV95-Koordinaten)
xmin, ymin, xmax, ymax = 2617260, 1130263, 2617640, 1130553

with rasterio.open("C:/git/Master-Thesis-GEOAI/data/ortho_4.tif") as src:
    window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
    clip = src.read(window=window)
    transform = src.window_transform(window)
    crs = src.crs

print("Clipped shape:", clip.shape)  # Bands x H x W
