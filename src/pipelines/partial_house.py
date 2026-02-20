import cv2
import numpy as np
from shapely.geometry import Point

from src.patches.extractor import extract_patch
from src.pipelines.base import Pipeline, PipelineResult
from src.sam.partial import run_sam_detect_all
from src.sam.refine import run_sam_stage


class PartialHousePipeline(Pipeline):

    name = "PARTIAL"

    def execute(self, ctx):

        # --------------------------------------------------
        # Extract larger context
        # --------------------------------------------------

        img_big, poly_px_big = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=4.0,
        )

        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)

        temp_big_path = ctx.sam_dir / f"bld_{ctx.building_id:07d}_partial_context.png"
        cv2.imwrite(str(temp_big_path), img_big)

        # --------------------------------------------------
        # Detect all candidate roofs
        # --------------------------------------------------

        candidates = run_sam_detect_all(
            img=img_big,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
        )

        if not candidates:
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=[],
                outside_pts=[],
                metadata={"stage": "no_masks_found"}
            )

        # --------------------------------------------------
        # Compute center of original footprint
        # --------------------------------------------------

        center = poly_px_big.centroid
        center_point = Point(center.x, center.y)

        # --------------------------------------------------
        #Pick candidate containing that center
        # --------------------------------------------------

        selected = None

        for poly in candidates:
            if poly.contains(center_point):
                selected = poly
                break

        if selected is None:
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=[],
                outside_pts=[],
                metadata={"stage": "no_candidate_contains_center"}
            )

        # --------------------------------------------------
        #  Prepare refinement inputs
        # --------------------------------------------------

        # use selected polygon as new "footprint"
        selected_center = selected.centroid
        inside = [[int(selected_center.x), int(selected_center.y)]]
        outside = []

        # --------------------------------------------------
        # Run refinement stage
        # --------------------------------------------------

        refined_polygon = run_sam_stage(
            img=img_big,
            raw_path=temp_big_path,
            poly_px=selected,
            inside=inside,
            outside=outside,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
            geom=ctx.geom,
            crs=ctx.crs,
            tiff_path=ctx.tiff_path,
            context=3.0,  # partial starts larger
        )

        # --------------------------------------------------
        #  Return result
        # --------------------------------------------------

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=refined_polygon,
            inside_pts=inside,
            outside_pts=outside,
            metadata={"stage": "discovery+refine"}
        )