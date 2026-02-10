# Pipeline Architecture Documentation

## Overview

This document describes the clean, formalized architecture for the building detection and segmentation pipeline.

## Architecture Diagram

```
PATCH EXTRACTION
      ↓
DECISION STAGE (MLLM)
      ↓
   ROUTING
      ↓
   ┌──────────────────────────────────┐
   │                                  │
   ↓                 ↓                ↓
FULL HOUSE      PARTIAL HOUSE    DISCOVERY
PIPELINE        PIPELINE         PIPELINE
   │                 │                │
   └─────────────────┴────────────────┘
                     ↓
              DATABASE WRITE
```

## Core Components

### 1. Decision Module (`src/pipeline/decision.py`)

**Purpose:** MLLM makes the core decision about house presence and completeness.

**Key class:**
- `HouseDecision`: Dataclass representing the single source of truth
  - `house_present`: bool - Whether any house exists in the polygon
  - `full_house`: bool | None - Whether polygon covers nearly all of the house
  - `reason`: str | None - Error description

**Key function:**
- `mlqa_decide(clean_patch: Path) -> HouseDecision`

### 2. Routing Module (`src/pipeline/routing.py`)

**Purpose:** Routes to appropriate pipeline based on MLLM decision.

**Key function:**
- `route_pipeline(decision: HouseDecision) -> str`
  - Returns: "FULL", "PARTIAL", or "DISCOVERY"

**Routing logic:**
```python
if not decision.house_present:
    return "DISCOVERY"
if decision.full_house:
    return "FULL"
return "PARTIAL"
```

### 3. Pipeline Modules

#### 🟢 Full House Pipeline (`src/pipeline/full_house_pipeline.py`)

**Goal:** Refine an already good footprint

**Characteristics:**
- Normal patch size
- MLQA-generated points (inside/outside)
- Tight bbox around footprint
- Iterative SAM refinement (standard mode)

**Function:** `full_house_pipeline(img, poly_px, paths, bid)`

#### 🟡 Partial House Pipeline (`src/pipeline/partial_house_pipeline.py`)

**Goal:** Recover complete house from incomplete polygon

**Key differences:**
- BIGGER patch (context=5)
- Escalated SAM mode (larger bbox, optional points)
- Focus on bbox-driven segmentation

**Function:** `partial_house_pipeline(row, gdf, paths, bid)`

#### 🔵 Discovery Pipeline (`src/pipeline/discovery_pipeline.py`)

**Goal:** Find all houses when footprint is completely wrong

**Characteristics:**
- No anchor polygon
- Multi-building detection
- Simplified MLQA prompt (optimized for 8b model)
- Exploratory search pattern

**Function:** `discovery_pipeline(img, paths, bid)`

### 4. Simplified Discovery Prompt

The discovery prompt has been simplified for better 8b model performance:

**Before:** Complex nested list structure with building objects
**After:** Simple flat structure with `building1_points`, `building2_points`, etc.

This makes it easier for the 8b model to:
- Count buildings (0 to 3)
- Place points for each building
- Return consistent JSON format

## Main Loop Flow

The refactored `src/main.py` now follows a clean, readable pattern:

```python
# 1. Extract patch
img, poly_px = extract_patch(...)

# 2. Create outputs
raw_path, clean_path, debug_path = create_patch_outputs(...)

# 3. DECISION STAGE
decision = mlqa_decide(clean_path)

# 4. ROUTING
pipeline = route_pipeline(decision)

# 5. PIPELINE EXECUTION
if pipeline == "FULL":
    qa, inside_pts, outside_pts = full_house_pipeline(...)
elif pipeline == "PARTIAL":
    qa, inside_pts, outside_pts, img_big, poly_big = partial_house_pipeline(...)
elif pipeline == "DISCOVERY":
    buildings_found, negative_pts, discovered_polygons = discovery_pipeline(...)

# 6. DATABASE WRITE
write_mlqa(record)
```

## Benefits

### ✅ Clean Separation of Concerns
- MLLM decides (decision stage)
- Routing determines pipeline (routing stage)
- Pipelines execute (execution stage)
- No mixed responsibilities

### ✅ Easy to Evolve
Want a new strategy? Just add:
```python
elif pipeline == "RUINS":
    ruins_pipeline(...)
```

### ✅ Debuggable
Each pipeline can independently:
- Dump points
- Save masks
- Create overlays

### ✅ Matches Human Reasoning
- "Is there a house?" → `house_present`
- "Is it complete?" → `full_house`
- "Do I refine, escalate, or search?" → routing

### ✅ Simplified 8b Model Prompt
The discovery prompt now uses a flat structure that's easier for smaller models to generate consistently.

## File Structure

```
src/
├── pipeline/
│   ├── __init__.py
│   ├── decision.py              # Decision dataclass and MLLM decision
│   ├── routing.py               # Pipeline routing logic
│   ├── full_house_pipeline.py   # Standard refinement pipeline
│   ├── partial_house_pipeline.py # Escalated recovery pipeline
│   └── discovery_pipeline.py    # Multi-building detection pipeline
├── mlqa/
│   ├── mlqa_client.py           # Main QA analysis
│   ├── point_client.py          # Point placement
│   └── discovery_client.py      # Discovery mode (simplified prompt)
├── sam/
│   ├── sam_client.py            # SAM model interface
│   └── sam_stage.py             # SAM execution stages
└── main.py                       # Main loop (now clean and readable)
```

## Next Steps (Optional Enhancements)

1. **Point Strategy Pattern**: Make point generators pluggable
   - `FullHousePointStrategy`
   - `PartialHousePointStrategy`
   - `DiscoveryPointStrategy`

2. **SAM Strategy Pattern**: Make SAM strategies pluggable
   - `StandardSAMStrategy`
   - `EscalatedSAMStrategy`
   - `DiscoverySAMStrategy`

3. **Pipeline Configuration**: Move parameters to config
   - Patch context sizes
   - Iteration limits
   - Bbox scales

4. **Pipeline Metrics**: Add logging and metrics
   - Success rates per pipeline
   - Average processing time
   - Point accuracy
