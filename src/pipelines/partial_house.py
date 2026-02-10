from src.pipelines.base import Pipeline
from src.patches.extractor import extract_patch
from src.mlqa.discovery_client import discover_all_houses
from src.sam.sam_client import run_sam_multi_building
import cv2

class PartialHousePipeline(Pipeline):
    name = "PARTIAL_MULTI"

    def execute(self, ctx):
        # 1. Enlarged patch
        img_big, _ = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=5,
        )
        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)

        # 2. Enumerate houses (bounded, contextual)
        result = discover_all_houses(ctx.clean_path)
        houses = result.get("buildings_found", [])
        negatives = result.get("negative_points", [])

        if not houses:
            return []

        # 3. SAM for each house
        return run_sam_multi_building(
            image_path=ctx.raw_path,
            buildings_data=houses,
            negative_pts=negatives,
        )
