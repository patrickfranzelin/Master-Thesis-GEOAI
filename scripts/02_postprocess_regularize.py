"""Convenience entrypoint for post-processing detected buildings.

Runs tree-occlusion cleanup followed by buildingregulariser footprint regularization,
and writes the output to a target PostGIS table.
"""

from src.postprocess.occlusion_regularize import main


if __name__ == "__main__":
    main()
