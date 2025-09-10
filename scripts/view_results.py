# --- PROJ/GDAL Pfade zuerst setzen (ganz oben!) ---
import os
CONDA = os.environ.get("CONDA_PREFIX", r"C:\Users\franz\miniconda3\envs\geoai")
os.environ.setdefault("PROJ_LIB",  fr"{CONDA}\Library\share\proj")
os.environ.setdefault("GDAL_DATA", fr"{CONDA}\Library\share\gdal")

# pyproj explizit auf den Datenordner zeigen (falls nötig)
try:
    from pyproj import datadir as _pd
    _pd.set_data_dir(os.environ["PROJ_LIB"])
except Exception:
    pass

import warnings
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt


# --- Pfade anpassen ---
GEOTIFF_PATH = r"D:\Masterarbeit\data\ortho_4.tif"
GPKG_PATH    = r"D:\Masterarbeit\outputs\buildings_sam_tiles.gpkg"
LAYER_NAME   = "buildings_sam"
OUT_PNG      = r"D:\Masterarbeit\outputs\overlay_preview.png"  # oder None, um nicht zu speichern

# --- Anzeige-Parameter ---
VECTOR_EDGEWIDTH = 1.2
VECTOR_ALPHA     = 0.7

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # 1) Raster öffnen
    with rasterio.open(GEOTIFF_PATH) as src:
        raster_crs = src.crs
        fig, ax = plt.subplots(figsize=(8, 8))
        show(src, ax=ax)  # zeigt RGB/Mehrband automatisch; sonst Band-Index übergeben

        # 2) Vektor (GPKG) laden
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
        if gdf.empty:
            print("⚠️ GPKG-Layer ist leer – nichts zu plotten.")
        else:
            # 3) CRS-Match (falls nötig)
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            # 4) Auf sinnvolle Ausdehnung zoomen (Vektor-Bounds ∩ Raster-Ausdehnung)
            #    – wenn du immer Vollbild Raster willst, kommentiere den Block einfach aus.
            try:
                vxmin, vymin, vxmax, vymax = gdf.total_bounds
                rxmin, rymin, rxmax, rymax = src.bounds
                xmin = max(vxmin, rxmin)
                ymin = max(vymin, rymin)
                xmax = min(vxmax, rxmax)
                ymax = min(vymax, rymax)
                if xmin < xmax and ymin < ymax:
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
            except Exception:
                pass

            # 5) Vektor-Overlay
            gdf.plot(ax=ax, facecolor="none", edgecolor="red",
                     linewidth=VECTOR_EDGEWIDTH, alpha=VECTOR_ALPHA)

        ax.set_title(f"Overlay: {os.path.basename(GEOTIFF_PATH)}  +  {LAYER_NAME}")
        ax.set_axis_off()
        plt.tight_layout()

        # 6) Optional als PNG speichern
        if OUT_PNG:
            os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
            plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
            print(f"✅ PNG gespeichert: {OUT_PNG}")

        plt.show()

if __name__ == "__main__":
    main()
