-- =========================================
-- Add full_house_present column to building_mlqa table
-- This column tracks whether the initial polygon covers the full building
-- or only a partial section, enabling better workflow separation
-- =========================================

BEGIN;

-- Add the column if it doesn't exist
ALTER TABLE src.building_mlqa 
ADD COLUMN IF NOT EXISTS full_house_present BOOLEAN DEFAULT NULL;

-- Add comment for documentation
COMMENT ON COLUMN src.building_mlqa.full_house_present IS 
'Indicates if the polygon covers nearly all of the house footprint. 
NULL = not analyzed, TRUE = full house, FALSE = partial house requiring escalation';

COMMIT;

-- Report
SELECT 
    COUNT(*) as total_records,
    COUNT(full_house_present) as analyzed_full_house,
    SUM(CASE WHEN full_house_present = true THEN 1 ELSE 0 END) as full_houses,
    SUM(CASE WHEN full_house_present = false THEN 1 ELSE 0 END) as partial_houses
FROM src.building_mlqa;
