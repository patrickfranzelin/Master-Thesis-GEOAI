# Discovery Mode: Multi-Building Detection

## Overview

Discovery mode is a new workflow that activates when `house_present = False`. Instead of skipping processing, the system now uses MLQA and SAM to discover and segment ALL buildings visible in the patch.

## Problem Statement

When the original polygon doesn't contain a house, the patch might still contain other buildings that should be detected and segmented. The old workflow simply skipped these cases, losing valuable building data.

## Solution

A three-stage discovery pipeline:

1. **MLQA Discovery**: Analyze the entire patch to find all buildings
2. **Point Generation**: Place positive points on each detected building
3. **Multi-SAM Segmentation**: Run SAM separately for each building with shared negative points

## Workflow

### 1. Detection Phase (MLQA Discovery)

When `house_present = False`, the system calls `discover_all_houses()` with a specialized MLQA prompt:

```python
discovery_result = discover_all_houses(clean_path)
```

Returns:
```json
{
  "buildings_found": [
    {
      "building_id": 1,
      "description": "rectangular metal roof, center-left",
      "inside_points": [[x1,y1], [x2,y2], [x3,y3]],
      "confidence": "high"
    },
    {
      "building_id": 2,
      "description": "mud compound, top-right",
      "inside_points": [[x1,y1], [x2,y2]],
      "confidence": "medium"
    }
  ],
  "negative_points": [[x1,y1], [x2,y2], ...],
  "total_buildings": 2
}
```

### 2. Segmentation Phase (Multi-Building SAM)

For each discovered building:
- Use its specific inside points
- Share common negative points across all buildings
- Run SAM independently for each building

```python
discovered_polygons = run_sam_discovery(
    img,
    raw_path,
    buildings_found,
    negative_pts,
    sam_dir,
    building_id
)
```

### 3. Visualization

Discovery mode creates special visualizations:
- `bld_XXXXXXX_discovery_overlay.png`: All detected buildings with colored polygons
- `bld_XXXXXXX_discovery_points.png`: Points used for detection

Each building is drawn in a different color and labeled (B1, B2, B3, etc.)

## Key Differences from Standard Workflow

| Aspect | Standard/Escalated | Discovery |
|--------|-------------------|-----------|
| **Trigger** | `house_present = True` | `house_present = False` |
| **Target** | Single building in polygon | All buildings in patch |
| **MLQA Prompt** | Analyze target building | Find all buildings |
| **Point Strategy** | 2-3 points on one roof | Multiple point sets, one per building |
| **SAM Calls** | 1 call with iterations | Multiple calls, one per building |
| **Output** | Single polygon | List of polygons |
| **Visualization** | Green outline | Multi-colored with labels |

## Use Cases

1. **Incorrect Polygons**: Dataset polygon is offset and misses the actual building
2. **Dense Areas**: Multiple buildings in one patch area
3. **Compound Structures**: Multiple connected or nearby buildings
4. **Edge Cases**: Building partially visible at patch edge

## Technical Details

### MLQA Discovery Client

File: `src/mlqa/discovery_client.py`

Key features:
- Specialized prompt for finding all buildings
- Instructs model to ignore original green polygon
- Generates distributed points for each building
- Provides confidence scores

### Multi-Building SAM

File: `src/sam/sam_client.py` - `run_sam_multi_building()`

Key features:
- Processes multiple buildings independently
- Shares negative points to avoid false positives
- Applies morphological cleanup to each mask
- Returns list of (mask, polygon) tuples

### Discovery SAM Stage

File: `src/sam/sam_stage.py` - `run_sam_discovery()`

Key features:
- Color-coded visualization for each building
- Labels buildings as B1, B2, B3, etc.
- Saves debug images showing all points
- Returns validated polygons only

## Database Storage

Discovery mode stores:
- `house_present`: False
- `full_house_present`: NULL (not applicable)
- `error_description`: "Discovery mode: found N buildings"
- `inside_pts`: Empty (not applicable)
- `outside_pts`: Shared negative points
- Additional context in error_description

## Example Output

```
Processing building 12345
Building 12345: No house in polygon - running DISCOVERY mode
  Discovery MLQA found 3 buildings in patch
  SAM mode: discovery - detecting 3 potential buildings
  Discovery mode: found 3 buildings
```

Generated files:
- `bld_0012345_discovery_overlay.png` - Shows 3 color-coded buildings
- `bld_0012345_discovery_points.png` - Shows all detection points

## Performance Considerations

- **Computational Cost**: Higher than standard mode (multiple SAM calls)
- **MLQA Tokens**: ~2x standard mode (more complex analysis)
- **Recommended**: Run on subset first to evaluate accuracy

## Future Enhancements

Potential improvements:

1. **Confidence Filtering**: Only segment high-confidence buildings
2. **Size Filtering**: Ignore very small detected objects
3. **Post-processing**: Merge nearby buildings or filter by shape
4. **Batch Processing**: Optimize multiple SAM calls
5. **Semantic Filtering**: Use building characteristics to filter false positives
6. **Spatial Clustering**: Group nearby buildings into compounds

## Configuration

Current settings in code:
- Max buildings detected: Unlimited (process all found)
- Morphological kernel: 7x7 (same as standard)
- Min confidence: None (process all)
- Point distribution: 2-3 per building

## Testing

To test discovery mode:
1. Find buildings with `house_present = False`
2. Check logs for "DISCOVERY mode" message
3. Review generated discovery visualization images
4. Verify multiple buildings are detected if present

## Comparison with Alternatives

### Why not SAM's Automatic Everything Mode?

SAM can automatically segment everything in an image without prompts. However:

**Pros of MLQA-guided approach:**
- More targeted and efficient
- Leverages semantic understanding of what is a building
- Provides interpretable results with descriptions
- Better control over what gets segmented

**Cons of pure automatic mode:**
- Segments everything (trees, roads, shadows, etc.)
- Requires extensive post-filtering
- Less efficient computationally
- No semantic understanding of results

## Conclusion

Discovery mode transforms `house_present = False` from a failure case into an opportunity to discover and segment multiple buildings. This increases the value extracted from each patch and improves overall dataset completeness.
