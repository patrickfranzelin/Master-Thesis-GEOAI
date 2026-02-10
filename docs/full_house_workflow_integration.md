# Full House Present Workflow Integration

## Overview

This document explains the improvements made to the patch generation and workflow separation for handling full vs partial houses.

## Changes Summary

### 1. Improved Patch Generation
**Problem**: Initial patches didn't capture full buildings when polygons were inaccurate.

**Solution**: 
- Increased default `context` from 2 to 3 (50% more coverage area)
- Increased escalation `context` from 4 to 5 for partial houses

### 2. Full House Present Field Integration
**Problem**: The `full_house_present` field was partially implemented but not well integrated.

**Solution**: Complete integration throughout the workflow.

## Workflow Separation

The system now has two distinct workflows:

### Standard Workflow (Full Houses)
**Triggered when**: `full_house_present = True`

**Behavior**:
- Uses the standard extracted patch (context=3)
- Uses MLQA-provided positive/negative points
- Uses smaller SAM bbox (scale=0.2)
- Logging: `"Building {id}: Full house detected - standard SAM workflow"`

### Escalated Workflow (Partial Houses)
**Triggered when**: `full_house_present = False`

**Behavior**:
- Re-extracts a larger patch (context=5)
- Resets MLQA points (doesn't use them)
- Uses larger SAM bbox (scale=0.8)
- Logging: `"Building {id}: Partial house detected - escalated SAM workflow"`

## Database Changes

### Migration Required
Run the SQL migration script to add the new column:

```bash
psql -U postgres -d geoai -f scripts/sql_script/add_full_house_present_column.sql
```

### New Column
- **Name**: `full_house_present`
- **Type**: `BOOLEAN`
- **Default**: `NULL`
- **Meaning**:
  - `NULL`: Not analyzed yet
  - `TRUE`: Polygon covers nearly all of the house
  - `FALSE`: Polygon only covers part of the house (clipped/incomplete)

## Code Changes

### src/patches/extractor.py
- Changed default `context` parameter from 2 to 3

### src/main.py
- Separated full house vs partial house logic into clear branches
- Added `full_house_present` to database record
- Improved logging for each workflow type

### src/db/writer.py
- Added `full_house_present` to INSERT statement
- Added `full_house_present` to UPDATE statement

### src/mlqa/mlqa_client.py
- Ensured `full_house_present` is always returned (even in error cases)

### src/sam/sam_stage.py
- Changed parameter from `big` (boolean) to `mode` (string)
- Added docstring with full parameter documentation
- Added detailed logging showing which mode is active
- Clear separation of behavior for each mode

## Usage Examples

### Querying Full vs Partial Houses

```sql
-- Count of each type
SELECT 
    full_house_present,
    COUNT(*) as count
FROM src.building_mlqa
WHERE house_present = true
GROUP BY full_house_present;

-- Find buildings that needed escalation
SELECT 
    building_id,
    error_description
FROM src.building_mlqa
WHERE house_present = true 
  AND full_house_present = false;

-- Success rate
SELECT 
    ROUND(100.0 * SUM(CASE WHEN full_house_present = true THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_full_houses
FROM src.building_mlqa
WHERE house_present = true;
```

### Interpreting Logs

Standard workflow:
```
Processing building 1234
Building 1234: Full house detected - standard SAM workflow
  SAM mode: standard (full house) - using MLQA points
```

Escalated workflow:
```
Processing building 5678
Building 5678: Partial house detected - escalated SAM workflow
  SAM mode: escalated (partial house) - using larger bbox, resetting MLQA points
```

## Benefits

1. **Better Initial Patches**: Increased context means fewer missed buildings
2. **Clear Code**: Easy to understand which workflow is being used
3. **Database Tracking**: Can analyze how often escalation is needed
4. **Better Debugging**: Clear logging shows workflow decisions
5. **Maintainability**: Separated concerns make future changes easier

## Future Enhancements

Potential areas for further improvement:

1. **Adaptive Context**: Could use building size to determine optimal context
2. **Different MLQA Prompts**: Could use different prompts for escalated cases
3. **SAM Parameters**: Could tune SAM parameters differently for each mode
4. **Multi-Level Escalation**: Could have more than two levels (e.g., context=3, 5, 7)
5. **Metrics Dashboard**: Track escalation frequency and success rates

## Testing

To test the changes:

1. Run the SQL migration script
2. Process a few buildings with the updated code
3. Check the logs to see workflow separation
4. Query the database to verify `full_house_present` is being stored
5. Verify that partial houses get larger patches and different SAM treatment
