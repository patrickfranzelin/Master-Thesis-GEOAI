import cv2

from src.mlqa.relocation_client import relocate_building
from src.patches.extractor import extract_patch
from src.pipelines.base import Pipeline, PipelineResult
from src.sam.partial import run_sam_detect_all


class PartialHousePipeline(Pipeline):

    name = "PARTIAL"

    def execute(self, ctx):

        # Enlarge context
        img_big, poly_px_big = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=4.0,
        )

        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)

        # Save enlarged patch
        temp_big_path = ctx.sam_dir / f"bld_{ctx.building_id:07d}_partial_context.png"
        cv2.imwrite(str(temp_big_path), img_big)

        # Get relocation points
        relocation_pts = relocate_building(temp_big_path)

        inside = relocation_pts.get("inside", [])
        outside = relocation_pts.get("outside", [])

        if not inside:
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=[],
                outside_pts=[],
                metadata={"stage": "discovery_failed"}
            )

        # Run automatic mask detection
        candidates = run_sam_detect_all(
            img=img_big,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
        )

        if not candidates:
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=inside,
                outside_pts=outside,
                metadata={"stage": "no_masks_found"}
            )

        # 🔥 RETURN ALL POLYGONS
        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=candidates,   # <- THIS IS THE FIX
            inside_pts=inside,
            outside_pts=outside,
            metadata={"stage": "discovery_only"}
        )