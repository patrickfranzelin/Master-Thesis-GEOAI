from src.pipelines.base import Pipeline, PipelineResult
from src.mlqa.point_client import analyze_points
from src.sam.refine import run_sam_stage


class FullHousePipeline(Pipeline):
    name = "FULL"

    def execute(self, ctx):
        pts = analyze_points(ctx.debug_path)
        inside = pts.get("inside", [])
        outside = pts.get("outside", [])

        sam_polygon = run_sam_stage(
            img=ctx.img,
            raw_path=ctx.raw_path,
            poly_px=ctx.poly_px,
            inside=inside,
            outside=outside,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
        )

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=sam_polygon,
            inside_pts=inside,
            outside_pts=outside,
            metadata={"mode": "standard"},
        )
