from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Optional

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
    # For discovery mode: enlarged patch for better context
    discovery_path: Optional[Path] = None
    discovery_img: Optional[np.ndarray] = None

