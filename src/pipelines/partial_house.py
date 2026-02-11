from src.pipelines.base import Pipeline, PipelineResult
from src.patches.extractor import extract_patch
from src.mlqa.discovery_client import discover_all_houses
from src.sam.multi import run_sam_multi_building
import cv2


class PartialHousePipeline(Pipeline):
    name = "PARTIAL_MULTI"

    def execute(self, ctx):

        # Extract enlarged patch
        img_big, _ = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=5,
        )
        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)

        # Save enlarged raw image
        discovery_raw_path = (
            ctx.sam_dir.parent
            / "raw"
            / f"bld_{ctx.building_id:07d}_discovery_raw.png"
        )

        cv2.imwrite(str(discovery_raw_path), img_big)

        # Store for main logging
        ctx.discovery_path = discovery_raw_path
        ctx.discovery_img = img_big

        #  Run discovery (MLQA enumeration)
        result = discover_all_houses(discovery_raw_path)

        buildings = result.get("buildings", [])
        outside_pts = result.get("negative_points", [])

        if not buildings:
            # No buildings detected in expanded context
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=[],
                inside_pts=[],
                outside_pts=outside_pts,
                metadata={
                    "buildings_found": 0,
                    "discovery_path": str(discovery_raw_path),
                },
            )

        #  Run SAM multi-building
        sam_results = run_sam_multi_building(
            image_path=discovery_raw_path,
            buildings_data=buildings,
            negative_pts=outside_pts,
        )

        # Extract only valid polygons
        polygons = [
            poly for mask, poly in sam_results
            if poly is not None
        ]

        # Collect all inside points used (for logging)
        all_inside_pts = []
        for b in buildings:
            all_inside_pts.extend(b.get("inside_points", []))

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=polygons,   # Always List[Polygon]
            inside_pts=all_inside_pts,
            outside_pts=outside_pts,
            metadata={
                "buildings_found": len(polygons),
                "discovery_path": str(discovery_raw_path),
            },
        )
