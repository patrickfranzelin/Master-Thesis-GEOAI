from src.pipelines.base import Pipeline
from src.mlqa.point_client import analyze_points
from src.sam.sam_stage import run_sam_stage

class FullHousePipeline(Pipeline):
    name = "FULL"

    def execute(self, ctx):
        pts = analyze_points(ctx.debug_path)
        inside = pts.get("inside", [])
        outside = pts.get("outside", [])

        return run_sam_stage(
            img=ctx.img,
            raw_path=ctx.raw_path,
            poly_px=ctx.poly_px,
            inside=inside,
            outside=outside,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
            mode="standard",
        )
