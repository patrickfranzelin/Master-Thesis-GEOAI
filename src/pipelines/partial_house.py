from src.pipelines.base import Pipeline, PipelineResult
from src.patches.extractor import extract_patch
from src.mlqa.discovery_client import discover_all_houses
from src.sam.multi import segment_multiple_buildings
import cv2


class PartialHousePipeline(Pipeline):
    name = "PARTIAL_MULTI"

    def execute(self, ctx):

        # 1️⃣ Extract enlarged patch
        img_big, _ = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=5,
        )

        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)

        # 2️⃣ Save raw discovery image (NO overlay)
        discovery_raw_path = (
            ctx.sam_dir.parent
            / "raw"
            / f"bld_{ctx.building_id:07d}_discovery_raw.png"
        )

        cv2.imwrite(str(discovery_raw_path), img_big)

        ctx.discovery_path = discovery_raw_path
        ctx.discovery_img = img_big

        # 3️⃣ Run simplified discovery
        result = discover_all_houses(discovery_raw_path)

        inside = result.get("inside", [])
        outside = result.get("outside", [])

        if not inside:
            # No buildings detected
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=[],
                inside_pts=[],
                outside_pts=outside,
                metadata={
                    "buildings_found": 0,
                    "discovery_path": str(discovery_raw_path),
                },
            )

        # 4️⃣ Convert flat inside list → building structure for SAM
        buildings_data = [
            {"inside_points": [pt]} for pt in inside
        ]

        # 5️⃣ Run SAM multi-building
        sam_results = segment_multiple_buildings(
            image_path=discovery_raw_path,
            buildings_data=buildings_data,
            negative_pts=outside,
        )

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=sam_results,
            inside_pts=inside,
            outside_pts=outside,
            metadata={
                "buildings_found": len(inside),
                "discovery_path": str(discovery_raw_path),
            },
        )
