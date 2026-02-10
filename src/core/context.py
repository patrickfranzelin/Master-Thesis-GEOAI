from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class PipelineContext:
    building_id: int
    img: np.ndarray
    poly_px: object
    raw_path: Path
    clean_path: Path
    debug_path: Path
    sam_dir: Path
    geom: object
    crs: object
    tiff_path: Path
