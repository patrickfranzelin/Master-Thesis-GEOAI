# Master Thesis GEOAI

Multimodal quality assessment and refinement of **automatically generated building footprints** using:
- a multimodal language model (MLLM) for semantic reasoning,
- SAM/SAM3 for geometry refinement,
- PostGIS for scalable geospatial data management.

This repository implements a two-stage, decoupled architecture:
1. **Evaluate** if a building footprint is valid and complete (MLLM-based QA).
2. **Refine** only footprints that can be improved (SAM-based segmentation).

---

## 1) Project Goal

Large regions (especially in parts of the Global South) have no reliable authoritative building reference data. Classical QA methods that compare to official datasets are often impossible there.

This project explores a **reference-free validation strategy**:
- inspect imagery + footprint together,
- classify quality semantically,
- route to specialized refinement pipelines,
- write improved polygons back to a geospatial database.

---

## 2) Method Overview

### System architecture

The workflow is modular and split into decision + correction:

- **MLLM stage** (semantic QA):
  - Is there a roof in the footprint?
  - Is the footprint mostly complete?
  - Which error type is visible?

- **SAM stage** (geometric correction):
  - Uses point prompts (+ optional bounding box / semantic prompt)
  - Produces refined binary masks
  - Converts masks back to polygons and geospatial CRS

Flowchart:

![Main workflow](Theory/flowchart_v1.png)

---

## 3) Processing Pipeline (End-to-End)

The implementation in `src/main.py` executes the following sequence:

1. **Load buildings from PostGIS + AOI filtering**
2. **Extract raster patches** around each building (`512×512`)
3. **Create patch variants** for different tasks:
   - raw patch (segmentation)
   - clean patch (polygon overlay for QA)
   - debug patch (centroid marker for point prompting)
4. **Run MLLM QA** and build a decision object:
   - `house_present`
   - `full_house_present`
   - `error_description`
5. **Route to pipeline**:
   - no house → stop and store QA result
   - full house → `FullHousePipeline`
   - partial/uncertain → `PartialHousePipeline`
6. **Run SAM refinement** (iterative; with optional context expansion)
7. **Backproject polygons** to geospatial coordinates and store in DB
8. **Store metadata** (prompt points, error info, run id, detection type, etc.)

---

## 4) Pipeline Logic

### A) FullHousePipeline (`src/pipelines/full_house.py`)
Used when the footprint already covers most of the building.

- generates positive/negative points from debug patch,
- refines with SAM,
- expands context if the prediction touches patch borders,
- outputs refined polygon(s) + metadata.

### B) PartialHousePipeline (`src/pipelines/partial_house.py`)
Used when footprint is partial or uncertain.

- extracts a larger context patch,
- runs SAM text/concept discovery (`house` / `building`) to find candidates,
- selects best candidate around footprint centroid,
- runs iterative point-guided refinement,
- returns final refined polygon + crop metadata for correct backprojection.

> If you have dedicated figures for `Theory/fullpipline` and `Theory/partialpipline`, place them in the repo and add them here; the current repository snapshot does not include those files.

---

## 5) Repository Structure

```text
src/
  main.py                     # Orchestrates complete workflow
  core/context.py             # Shared pipeline context object

  pipelines/
    router.py                 # Decision routing to full/partial pipelines
    full_house.py             # Standard refinement path
    partial_house.py          # Discovery + refinement path
    base.py                   # Common Pipeline / PipelineResult

  mlqa/
    mlqa_client.py            # QA prompts + structured MLLM parsing
    point_client.py           # Positive/negative point generation
    decision.py               # Converts QA JSON into routing decision

  patches/
    extractor.py              # Geo patch extraction + pixel transforms
    create_patch_output.py    # raw/clean/debug image generation

  sam/
    model_sam3.py             # SAM3 wrappers (points / text / exemplar)
    refine.py                 # Iterative SAM refinement stage
    partial.py                # Detect-all candidate discovery
    occlusion.py              # Tree/occlusion segmentation helpers

  db/
    writer.py                 # Persist QA + refined geometries
    loader.py                 # DB loading utilities
    export_to_filegdb.py      # Export helpers
```

---

## 6) Inputs and Outputs

### Required Inputs

1. **PostgreSQL/PostGIS tables** (expected schemas under `src.*` namespace)
   - building geometries (`src.buildings`)
   - AOI polygon (`src.aoi`)
2. **GeoTIFF paths** for each building row (`tiff_path` column)
3. **Runtime model services / weights**
   - MLLM endpoint (RunPod/OpenAI-compatible API)
   - SAM3 weights in `models/sam3_weights/sam3.pt`

### Runtime Environment Variables

- `PG_CONN` → SQLAlchemy/PostGIS connection string
- `RUNPOD_ID` → host id used to call the multimodal endpoint

### Main Outputs

- patch artifacts in `outputs/db_results/{raw,clean,debug,sam,comparison}`
- QA decisions written to `src.building_mlqa`
- refined footprints written to `src.detected_house`
- optional tree masks/polygons written via writer utilities

---

## 7) Installation

This project uses Python `>=3.11` and is configured with `pyproject.toml`.

```bash
# from repository root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

---

## 8) Run the Pipeline

```bash
export PG_CONN="postgresql+psycopg2://USER:PASS@HOST:5432/DB"
export RUNPOD_ID="<your-runpod-id>"
python -m src.main
```

Notes:
- `AOI_ID` is currently set in `src/main.py`.
- The processing limit is currently set to `LIMIT 10` for testing.
- Adjust context factors and prompt behavior in pipeline files for your study area.

---

## 9) Current Status and Thesis Context

This repository is a research implementation for a master thesis workflow and proof-of-concept. It is optimized for transparent experimentation, modularity, and reproducibility rather than turnkey production deployment.

Planned documentation extensions:
- evaluation metrics (IoU, boundary quality, error classes),
- baseline comparisons,
- ablation studies for point prompts and context expansion,
- additional figures for full/partial pipeline internals.

---

## 10) Example Visual Outputs

Current examples in this repository:

| Raw Extraction | MLLM QA / Prompting | Refined Result |
|---|---|---|
| ![raw-1](Theory/img.png) | ![qa-1](Theory/img_2.png) | ![refined-1](Theory/img_1.png) |
| ![raw-2](Theory/img_4.png) | ![qa-2](Theory/img_5.png) | ![refined-2](Theory/img_3.png) |
| ![raw-3](Theory/img_6.png) | ![qa-3](Theory/poly_4_points.png) | ![refined-3](Theory/img_7.png) |

---

## 11) Citation

If you use this code in academic work, please cite your thesis and the key model/data sources (Open Buildings, SAM, and the selected MLLM backend).
