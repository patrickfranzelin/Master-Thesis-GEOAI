from src.pipelines.base import Pipeline, PipelineResult
from src.mlqa.point_client import analyze_points
from src.sam.occlusion import segment_trees
from src.sam.refine import run_sam_stage
from src.patches.extractor import extract_patch
import cv2


class FullHousePipeline(Pipeline):
    name = "FULL"

    def execute(self, ctx):

        pts = analyze_points(ctx.debug_path)
        inside_base = pts.get("inside", [])
        outside_base = pts.get("outside", [])

        context_refine = 1.5
        max_expand = 3
        refined_polygon = None

        for expand_iter in range(max_expand):

            print(f"FULL expansion iteration {expand_iter + 1}")

            # ---------------------------------------------
            # Re-extract patch with larger context
            # ---------------------------------------------
            img, poly_px, win = extract_patch(
                ctx.geom,
                ctx.crs,
                ctx.tiff_path,
                context=context_refine,
            )

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            temp_raw_path = ctx.sam_dir / f"bld_{ctx.building_id:07d}_full_refine.png"
            cv2.imwrite(str(temp_raw_path), img)

            # Important: recompute inside/outside relative to new patch
            inside = inside_base.copy()
            outside = outside_base.copy()

            result = run_sam_stage(
                img=img,
                raw_path=temp_raw_path,
                poly_px=poly_px,
                inside=inside,
                outside=outside,
                out_dir=ctx.sam_dir,
                bid=ctx.building_id,
            )

            if result == "EXPAND_PATCH":
                context_refine *= 1.5
                print(f"🔁 FULL expanding patch → new context: {context_refine:.2f}")
                continue

            if result is None:
                refined_polygon = None
                break

            refined_polygon = result
            tree_masks, tree_polys = segment_trees(temp_raw_path)
            sam_size = img.shape[0]
            break

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=refined_polygon,
            inside_pts=inside_base,
            outside_pts=outside_base,
            metadata={
                "mode": "standard",
                "context_used": context_refine,
                "win": win,
                "sam_input_size": img.shape[0],
                "tree_polygons": tree_polys,
            },
        )