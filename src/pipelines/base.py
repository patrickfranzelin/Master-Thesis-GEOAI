from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class PipelineResult:
    """
    Standard result format for all pipelines.
    
    Attributes:
        pipeline_name: Name of the pipeline that executed
        sam_polygons: List of SAM-generated polygons (or single polygon for full house)
        inside_pts: Points used as positive prompts for SAM
        outside_pts: Points used as negative prompts for SAM
        metadata: Additional pipeline-specific data
    """
    pipeline_name: str
    sam_polygons: Any  # Can be single polygon, list of polygons, or None
    inside_pts: list
    outside_pts: list
    metadata: Optional[dict] = None


class Pipeline:
    name: str

    def execute(self, ctx):
        """
        Execute the pipeline and return PipelineResult.
        
        Returns:
            PipelineResult with SAM outputs and point data
        """
        raise NotImplementedError
