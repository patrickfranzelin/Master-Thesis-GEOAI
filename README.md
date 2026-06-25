# GeoAI Building Footprint Refinement

Multimodal quality assessment and segmentation-based refinement of building footprints for the master thesis **Integrating Vision and Multimodal Language Models for Collaborative Building Footprint Quality Assessment**.

The project evaluates and improves existing building polygons rather than generating a dataset from scratch. It combines a multimodal vision-language model for quality assessment, SAM-based segmentation for geometric correction, PostGIS for geospatial storage, and manual/statistical evaluation for thesis reporting.

![Workflow overview](docs/readme_assets/workflow_overview.jpg)

## What This Repository Does

The pipeline starts from building footprints, aerial image tiles, and AOI polygons stored in PostGIS. For each building it:

1. extracts an image patch around the footprint,
2. asks a multimodal LLM whether a roof is present and whether the polygon covers the full roof,
3. routes the case to a full-house, partial-house, or rejection path,
4. prompts SAM with VLM-generated points and footprint-derived boxes,
5. back-projects SAM polygons from pixel space to WGS84,
6. stores raw detections, tree detections, and regularized polygons in PostGIS,
7. exports FileGDB results and evaluation data.

The thesis motivation is that global AI-generated datasets such as Google Open Buildings provide broad coverage but still contain geometric errors: shifted footprints, oversimplified outlines, missing roof parts, and false positives. The implemented workflow tests whether multimodal models can detect these errors and whether their feedback can be converted into useful spatial refinements.

## Results Snapshot

The current thesis evaluation is based on the `src_google` database schema and generated statistics under `evaluation/statistics/`.

| Metric | Result |
| --- | ---: |
| Manual evaluation cases | 1,263 |
| Original Google footprints rated bad or ok | 98.5% |
| Postprocessed footprints rated good or perfect | 77.8% |
| Improved cases | 1,040 / 1,256 (82.8%) |
| Degraded cases | 52 / 1,256 (4.1%) |
| Average manual score before/after | 1.55 -> 3.23 |

| Country | n | Improved | Degraded | Good or perfect after postprocessing | Median post shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Liberia | 380 | 86.1% | 1.3% | 79.5% | 3.1 m |
| Mexico | 261 | 77.0% | 5.7% | 72.4% | 3.7 m |
| Mozambique | 167 | 86.2% | 3.0% | 83.8% | 3.7 m |
| Nepal | 192 | 76.0% | 8.3% | 65.6% | 5.1 m |
| Niger | 256 | 86.7% | 4.3% | 85.9% | 1.7 m |

![Country improvement rates](docs/readme_assets/country_improvement_rates.png)

![Stage comparison](docs/readme_assets/stage_comparison.png)

![Postprocessing shift vectors by country](docs/readme_assets/shift_vectors_by_country.png)

Important limitations found in the evaluation:

- SAM often creates detailed masks with many vertices; regularization reduces complexity but can change area.
- Tree occlusion and low roof-background contrast remain difficult.
- The semantic error descriptions are useful but inconsistent: in a 102-row semantic review, 11 latest descriptions were correct, 25 partly correct, 37 too vague, and 26 wrong.
- MLQA detected shifted buildings with high recall, but over-predicted several shape-related error categories.

## Example Outputs

### Full-House Refinement

Used when the input footprint already covers most of the visible roof.

| Input footprint | Refined SAM result |
| --- | --- |
| ![Full-house input](docs/readme_assets/full_input.png) | ![Full-house result](docs/readme_assets/full_result.png) |

### Partial-House Refinement

Used when the input footprint covers only part of a roof. The pipeline first detects candidate roof structures in a larger context patch, then refines the selected candidate.

| Input footprint | Candidate discovery | Refined result |
| --- | --- | --- |
| ![Partial-house input](docs/readme_assets/partial_input.png) | ![Partial-house discovery](docs/readme_assets/partial_discovery.png) | ![Partial-house result](docs/readme_assets/partial_result.png) |

## Pipeline Architecture

```mermaid
flowchart TD
    A[PostGIS buildings + AOI + TIFF links] --> B[Patch extraction]
    B --> C[MLLM presence and coverage QA]
    C --> D{Routing}
    D -->|full roof visible| E[FullHousePipeline]
    D -->|partial roof visible| F[PartialHousePipeline]
    D -->|no roof| G[Record QA result only]
    E --> H[VLM point prompts]
    F --> I[SAM candidate discovery]
    I --> H
    H --> J[Iterative SAM refinement]
    J --> K[Tree detection]
    K --> L[Back-project to WGS84]
    L --> M[PostGIS detected_house and detected_tree]
    M --> N[Postprocess regularization]
    N --> O[Manual/statistical evaluation]
```

### Main Components

| Component | Files | Role |
| --- | --- | --- |
| Orchestration | `src/main.py` | Loads AOI/buildings, runs MLQA, dispatches pipelines, stores results, exports FileGDB. |
| Patch extraction | `src/patches/extractor.py` | Converts WGS84/vector footprints into raster windows and SAM pixel coordinates. |
| MLQA | `src/mlqa/*.py` | Calls Qwen3-VL via an OpenAI-compatible RunPod endpoint for presence, coverage, error tags, and point prompts. |
| Routing | `src/pipelines/router.py` | Chooses full-house or partial-house processing from the MLQA decision. |
| SAM refinement | `src/sam/*.py` | Runs prompt-based segmentation, auto-mask discovery, text-based tree detection, and mask-to-polygon conversion. |
| Postprocessing | `src/postprocess/occlusion_regularize.py`, `src/postprocess/occlusion/` | Merges overlapping outputs, accounts for tree occlusion, and regularizes building geometry. |
| Database I/O | `src/db/*.py`, `scripts/sql_script/` | Imports source data, writes MLQA/SAM outputs, and exports FileGDB layers. |
| Evaluation | `evaluation/`, `evaluation/statistics/` | Manual review app, semantic review app, contrast metrics, thesis tables, and figures. |

## Database State

The local PostGIS database inspected on 2026-06-25 contains two active schemas:

| Schema | Purpose |
| --- | --- |
| `src` | Smaller/current development schema. |
| `src_google` | Thesis-scale Google Open Buildings experiment and evaluation schema. |

Current table counts:

| Table | `src` rows | `src_google` rows |
| --- | ---: | ---: |
| `aoi` | 3 | 7 |
| `buildings` | 1,011 | 8,269 |
| `building_mlqa` | 459 | 3,711 |
| `detected_house` | 430 | 8,608 |
| `detected_house_regularized` | 426 | 6,767 |
| `detected_tree` | 5,887 | 8,490 |
| `evaluation` | - | 1,263 |
| `semantic_description_evaluation` | - | 102 |
| `tiffs` | 1 | 1 |

`src_google.detected_house` currently contains:

| Detection type | Rows |
| --- | ---: |
| `global_discovery` | 6,586 |
| `partial` | 1,284 |
| `full` | 738 |

Main geometry columns use EPSG:4326. Building inputs are `MULTIPOLYGON`; detected houses, regularized houses, trees, and TIFF extents are `POLYGON`.

## Repository Layout

```text
src/
  core/                 Pipeline context objects
  db/                   PostGIS loading, writing, and FileGDB export
  mlqa/                 VLM prompts and JSON parsing
  patches/              Raster patch extraction and patch visualizations
  pipelines/            Full, partial, and global discovery workflows
  postprocess/          Matching, deduplication, occlusion repair, regularization
  sam/                  SAM model wrappers and mask-to-polygon utilities
  utils/                Geometry, drawing, rendering, and IO helpers

evaluation/
  server.py             Manual evaluation API
  semantic_server.py    Semantic description evaluation API
  statistics/           Thesis tables, figures, notebooks, and generators

scripts/
  sql_script/           Source data import and schema helpers
  02_postprocess_regularize.py
  db_copy_local_to_docker.ps1

docs/
  SOURCE_CODE_DOCUMENTATION.md
  docker_postgis.md
  readme_assets/
```

## Setup

Python 3.11 is recommended.

```powershell
uv sync
```

Required runtime services:

- PostgreSQL with PostGIS
- a local or remote OpenAI-compatible VLM endpoint
- source GeoTIFF and building/AOI data referenced by the database

Environment variables:

```powershell
$env:PG_CONN="postgresql://USER:PASSWORD@HOST:PORT/DB"
$env:RUNPOD_ID="<runpod-instance-id>"
```

The VLM client currently uses:

```text
https://<RUNPOD_ID>-7860.proxy.runpod.net/v1
model: qwen3vl8b
```

Do not commit `.env` files or database dumps. The repository intentionally ignores local data, outputs, virtual environments, Docker database backups, and local Docker env files.

## Running

Run the main processing pipeline:

```powershell
$env:PG_CONN="postgresql://USER:PASSWORD@localhost:5432/geoai"
$env:RUNPOD_ID="<runpod-instance-id>"
uv run python src/main.py
```

Run postprocessing regularization:

```powershell
uv run python scripts/02_postprocess_regularize.py --run-id "<run-id>"
```

Start the manual evaluation app:

```powershell
uv run python evaluation/server.py
```

Generate thesis statistics:

```powershell
uv run python evaluation/statistics/generate_thesis_statistics.py
```

## Docker PostGIS Option

The repo includes a Docker Compose setup for an isolated PostGIS copy. This is useful when you want to keep the local database untouched and work against a containerized copy.

```powershell
docker compose up -d db
.\scripts\db_copy_local_to_docker.ps1 -SourcePassword "<local-db-password>"
$env:PG_CONN="postgresql://postgres:geoai@localhost:5433/geoai"
```

See `docs/docker_postgis.md` for details.

## Data And Outputs

The repository does not track raw geospatial data or generated outputs. Expected local paths include:

- `data/` for country AOIs, GeoTIFFs, and source building files
- `outputs/db_results/` for pipeline output patches and FileGDB exports
- `evaluation/statistics/tables/` for generated result tables
- `evaluation/statistics/figures/` for generated result figures

## Research Questions

The codebase supports three thesis questions:

1. How accurately can MLLMs detect and describe geometric errors in building polygons?
2. How effective is the conversion of MLLM feedback into spatial refinements?
3. How well does the workflow improve Google Open Buildings footprints across different regions?

The current evidence suggests the pipeline substantially improves many visibly shifted or incomplete footprints, especially after regularization, but semantic error classification and difficult image contexts still require careful evaluation.
