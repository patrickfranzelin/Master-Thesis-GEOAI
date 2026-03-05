import cv2
from shapely.geometry import Point

from src.mlqa.point_client import analyze_points
from src.patches.extractor import extract_patch, extract_patch_pixel
from src.pipelines.base import Pipeline, PipelineResult
from src.sam.occlusion import segment_trees
from src.sam.partial import run_sam_detect_all
from src.sam.refine import run_sam_stage

PARTIAL_CONTEXT_START = 4.0


class PartialHousePipeline(Pipeline):
    name = "PARTIAL"

    def execute(self, ctx):

        # --------------------------------------------------
        # 1. Extract larger context patch for detection
        # --------------------------------------------------
        img_big, poly_px_big, win_big = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=PARTIAL_CONTEXT_START,
        )

        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)
        temp_big_path = ctx.sam_dir / f"bld_{ctx.building_id:07d}_partial_context.png"
        cv2.imwrite(str(temp_big_path), img_big)

        # --------------------------------------------------
        # 2. Detect all candidate roofs with SAM auto-mask
        # --------------------------------------------------
        candidates = run_sam_detect_all(
            img=img_big,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
        )

        if not candidates:
            print(" ✗ PARTIAL: no candidates detected")
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=[],
                outside_pts=[],
                metadata={"stage": "no_masks_found", "win": win_big},
            )

        # --------------------------------------------------
        # 3. Pick the candidate that contains the footprint centroid
        # --------------------------------------------------
        center_point = Point(poly_px_big.centroid.x, poly_px_big.centroid.y)
        selected = None
        for poly in candidates:
            if poly.contains(center_point):
                selected = poly
                break

        if selected is None:
            print(" ⚠ No candidate contains footprint center — picking nearest by centroid distance")
            selected = min(candidates, key=lambda p: p.centroid.distance(center_point))

        # --------------------------------------------------
        # 4. Refine with expanding context loop
        # --------------------------------------------------
        context_refine = 1.5
        max_expand = 3
        refined_polygon = None
        inside = []
        outside = []
        crop_info = None
        refine_img = None
        tree_polys = []

        for expand_iter in range(max_expand):
            print(f"SAM expansion iteration {expand_iter + 1}")

            refine_img, refine_poly_px, crop_info = extract_patch_pixel(
                img_big,
                selected,
                out_size=512,
                context=context_refine,
            )

            temp_refine_path = ctx.sam_dir / f"bld_{ctx.building_id:07d}_partial_refine.png"
            cv2.imwrite(str(temp_refine_path), refine_img)

            pts = analyze_points(temp_refine_path)
            inside = pts["inside"]
            outside = pts["outside"]

            result = run_sam_stage(
                img=refine_img,
                raw_path=temp_refine_path,
                poly_px=refine_poly_px,
                inside=inside,
                outside=outside,
                out_dir=ctx.sam_dir,
                bid=ctx.building_id,
            )

            if result == "EXPAND_PATCH":
                context_refine *= 1.5
                print(f"Expanding patch → new context: {context_refine:.2f}")
                continue

            if result is None:
                refined_polygon = None
                break

            refined_polygon = result
            tree_masks, tree_polys = segment_trees(temp_refine_path)
            break
        else:
            print(f" ✗ PARTIAL: patch expansion exhausted for building {ctx.building_id}")
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=inside,
                outside_pts=outside,
                metadata={
                    "stage": "max_expand_reached",
                    "context_used": context_refine,
                    "win": win_big,
                    "crop_info": crop_info,
                    "sam_input_size": refine_img.shape[0] if refine_img is not None else None,
                    "tree_polygons": [],
                },
            )

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=refined_polygon,
            inside_pts=inside,
            outside_pts=outside,
            metadata={
                "stage": "discovery+refine",
                "context_used": context_refine,
                "win": win_big,
                "crop_info": crop_info,
                "sam_input_size": refine_img.shape[0] if refine_img is not None else None,
                "tree_polygons": tree_polys,
            },
        )
