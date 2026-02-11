from src.pipelines.base import Pipeline, PipelineResult
from src.mlqa.point_client import analyze_points
from src.sam.refine  import run_sam_stage

class FullHousePipeline(Pipeline):
    name = "FULL"

    def execute(self, ctx):
        # Get points from MLQA
        pts = analyze_points(ctx.debug_path)
        inside = pts.get("inside", [])
        outside = pts.get("outside", [])

        # Run SAM
        sam_polygon = run_sam_stage(
            img=ctx.img,
            raw_path=ctx.raw_path,
            poly_px=ctx.poly_px,
            inside=inside,
            outside=outside,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
            mode="standard",
        )
        
        # Return structured result
        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=sam_polygon,
            inside_pts=inside,
            outside_pts=outside,
            metadata={"mode": "standard"}
        )
