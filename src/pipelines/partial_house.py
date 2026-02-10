from src.pipelines.base import Pipeline, PipelineResult
from src.patches.extractor import extract_patch
from src.mlqa.discovery_client import discover_all_houses
from src.sam.sam_client import run_sam_multi_building
from src.patches.create_patch_output import create_patch_outputs
import cv2

class PartialHousePipeline(Pipeline):
    name = "PARTIAL_MULTI"

    def execute(self, ctx):
        # 1. Extract enlarged patch for better context in discovery
        img_big, poly_px_big = extract_patch(
            ctx.geom,
            ctx.crs,
            ctx.tiff_path,
            context=5,
        )
        img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)
        
        # 2. Create discovery image with polygon overlay and save to disk
        # MLQA needs to see the enlarged context, not the original clean image
        out_dirs = {
            "raw": ctx.sam_dir.parent / "raw",
            "clean": ctx.sam_dir.parent / "clean",
            "debug": ctx.sam_dir.parent / "debug",
        }
        
        # Create discovery image with polygon overlay on enlarged patch
        discovery_img = img_big.copy()
        from src.utils.rendering import add_polygon_overlay
        discovery_img = add_polygon_overlay(discovery_img, poly_px_big)
        
        # Save discovery image
        discovery_path = ctx.sam_dir.parent / "clean" / f"bld_{ctx.building_id:07d}_discovery.png"
        cv2.imwrite(str(discovery_path), discovery_img)
        
        # Store in context for consistency
        ctx.discovery_path = discovery_path
        ctx.discovery_img = img_big

        # 3. Enumerate houses using the enlarged patch (fixes Bug 1)
        result = discover_all_houses(discovery_path)
        houses = result.get("buildings_found", [])
        negatives = result.get("negative_points", [])

        if not houses:
            # No buildings found in discovery mode
            return PipelineResult(
                pipeline_name=self.name,
                sam_polygons=[],
                inside_pts=[],
                outside_pts=negatives,
                metadata={
                    "buildings_found": 0,
                    "discovery_path": str(discovery_path)
                }
            )

        # 4. SAM must use the same discovery image (fixes Bug 2)
        # Use the raw enlarged image (no overlay) for SAM
        discovery_raw_path = ctx.sam_dir.parent / "raw" / f"bld_{ctx.building_id:07d}_discovery_raw.png"
        cv2.imwrite(str(discovery_raw_path), img_big)
        
        sam_results = run_sam_multi_building(
            image_path=discovery_raw_path,
            buildings_data=houses,
            negative_pts=negatives,
        )
        
        # Collect all inside points from all buildings
        all_inside_pts = []
        for building in houses:
            all_inside_pts.extend(building.get("inside_points", []))
        
        # Return structured result
        return PipelineResult(
            pipeline_name=self.name,
            sam_polygons=sam_results,
            inside_pts=all_inside_pts,
            outside_pts=negatives,
            metadata={
                "buildings_found": len(houses),
                "discovery_path": str(discovery_path)
            }
        )
