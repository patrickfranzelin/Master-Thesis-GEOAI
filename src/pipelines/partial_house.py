from src.mlqa.relocation_client import relocate_building
from src.sam.refine import run_sam_stage

class RelocationPipeline(Pipeline):
    name = "RELOCATION"

    def execute(self, ctx):

        img_big, poly_px_big = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=4,
        )

        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)

        relocation_pts = relocate_building(ctx.clean_path)

        inside = relocation_pts.get("inside", [])
        outside = relocation_pts.get("outside", [])

        if not inside:
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=None,
                inside_pts=[],
                outside_pts=[],
                metadata={"reason": "no relocation points"}
            )

        sam_poly = run_sam_stage(
            img=img_big,
            raw_path=ctx.raw_path,
            poly_px=poly_px_big,
            inside=inside,
            outside=outside,
            out_dir=ctx.sam_dir,
            bid=ctx.building_id,
            mode="escalated",
        )

        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=sam_poly,
            inside_pts=inside,
            outside_pts=outside,
            metadata={"mode": "relocation"}
        )
