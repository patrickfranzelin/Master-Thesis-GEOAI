# GeoAI Building Footprint Refinement
### Multimodal Quality Assessment and Segmentation of Building Footprints

This repository contains the implementation developed for the **Master Thesis: GeoAI Building Footprint Refinement**.

The system introduces a **multimodal geospatial AI workflow** that automatically:

1. evaluates building footprint quality
2. detects geometric errors
3. refines building polygons using segmentation models

The pipeline combines:

- **Multimodal Large Language Models (MLLMs)** for semantic reasoning
- **Segment Anything Model 3 (SAM3)** for segmentation
- **PostGIS** for geospatial data storage
- **Python GeoAI pipeline orchestration**

The approach allows **quality assessment without reference datasets**, which is particularly relevant for **data-scarce regions**.

---

# Pipeline Overview

The system processes building footprints in **seven stages**.

![Pipeline Overview](Theory/flowcharsamstage.png)

### Workflow

1. Load building footprints from PostGIS  
2. Extract aerial image patches  
3. Evaluate footprint quality using an MLLM  
4. Route buildings into refinement pipelines  
5. Generate segmentation prompts  
6. Refine geometry using SAM  
7. Write refined footprints to the database  

---

# Example Refinement — Full Building Pipeline

Example where the building footprint already captures most of the structure.

## Input Footprint

![Clean Patch](Theory/fullpipline/bld_0141916_clean.png)

The green polygon represents the **original building footprint**.

---

## Prompt Generation

Green points = positive roof prompts  
Red points = negative background prompts  
Yellow box = bounding box derived from the footprint  

![Prompt Input](Theory/fullpipline/bld_0141916_sam_input.png)

These prompts guide the segmentation model.

---

## Final SAM Segmentation

![Refined Polygon](Theory/fullpipline/bld_0141916_sam.png)

The segmentation mask is converted into a **refined building polygon**.

---

# Example Refinement — Partial Building Pipeline

Example where the original footprint captures **only part of the building**.

## Incomplete Footprint

![Partial Footprint](Theory/partialpipline/bld_1683655_clean.png)

The polygon only covers a small section of the visible roof.

---

## SAM Discovery Step

The system first detects **all candidate roof structures** in the larger context patch.

![Detected Structures](Theory/partialpipline/bld_1683655_detect_all_overlay.png)

This step uses a **text prompt segmentation**:

```
prompt: "house"
```

---

## Prompt-Based Refinement

Once the correct building candidate is selected, prompts are generated for refinement.

![Prompt Generation](Theory/partialpipline/bld_1683655_sam_input.png)

---

## Final Refined Building

![Final Polygon](Theory/partialpipline/bld_1683655_sam.png)

The result is a **corrected building footprint** matching the visible roof.

---

# Repository Structure

```
src/
│
├── main.py
│
├── patches/
│   ├── extractor.py
│   └── create_patch_output.py
│
├── mlqa/
│   ├── mlqa_client.py
│   ├── decision.py
│   └── point_client.py
│
├── pipelines/
│   ├── router.py
│   ├── full_house.py
│   ├── partial_house.py
│   └── base.py
│
├── sam/
│   ├── refine.py
│   ├── partial.py
│   ├── occlusion.py
│   └── model_sam3.py
│
└── postprocess/
    └── occlusion_regularize.py
```

---

# Core Modules

## Patch Extraction

Extracts **512×512 aerial patches** around building footprints.

File:

```
src/patches/extractor.py
```

Functions include:

- CRS transformation
- raster window extraction
- polygon → pixel coordinate conversion

---

## MLLM Quality Assessment

The multimodal model evaluates building footprints using aerial imagery.

File:

```
src/mlqa/mlqa_client.py
```

The model determines:

```
house_present
full_house_present
error_description
```

Possible error types:

- UNDERSEGMENTATION
- OVERSEGMENTATION
- MISALIGNMENT
- PARTIAL_VISIBILITY

---

## Pipeline Router

Routes buildings into processing pipelines.

File:

```
src/pipelines/router.py
```

Routing logic:

| Condition | Pipeline |
|---|---|
| no house detected | reject |
| full building | FullHousePipeline |
| partial building | PartialHousePipeline |

---

# FullHousePipeline

Used when the footprint already captures most of the building.

Steps:

1. Generate segmentation prompts
2. Create bounding box from footprint
3. Extract refinement patch
4. Run SAM segmentation
5. Expand context if segmentation touches patch border

Implementation:

```
src/pipelines/full_house.py
```

---

# PartialHousePipeline

Used when the footprint only captures **part of a building**.

Steps:

1. Extract large context patch
2. Detect all candidate roofs using SAM
3. Select candidate containing the footprint centroid
4. Extract sub-patch
5. Generate prompts
6. Run SAM refinement

Implementation:

```
src/pipelines/partial_house.py
```

---

# Segmentation Model

Segmentation is performed using **SAM3**.

Two segmentation modes are used:

### Visual Prompt Segmentation

Input prompts:

```
inside points
outside points
bounding box
```

### Text Prompt Segmentation

```
prompt: "house"
```

Used in the **discovery stage**.

---

# Iterative Refinement

The segmentation stage uses an **iterative feedback loop**.

Algorithm:

```
1 run SAM segmentation
2 compute bounding box of mask
3 re-run SAM with tighter bbox
4 repeat up to 3 iterations
```

If the predicted mask touches the patch boundary:

```
expand patch
rerun segmentation
```

---

# Database Integration

The system uses **PostgreSQL + PostGIS**.

Input tables:

```
src.buildings
src.aoi
```

Output tables:

```
src.detected_house
src.detected_tree
src.detected_house_regularized
```

Results can optionally be exported to **FileGDB**.

---

# Running the Pipeline

## Requirements

Python ≥ 3.10

Main libraries:

```
torch
opencv-python
rasterio
geopandas
shapely
ultralytics
sqlalchemy
openai
```

---

## Environment variables

```
PG_CONN=postgresql://user:password@host/database
RUNPOD_ID=<runpod instance>
```

---

## Execute pipeline

```
python src/main.py
```

---

# Research Context

This repository accompanies the thesis:

**Multimodal GeoAI for Automated Quality Assessment and Refinement of Building Footprints**

Key research questions:

- Can MLLMs evaluate geospatial geometry quality?
- Can segmentation models refine building footprints automatically?
- Can such systems operate **without reference datasets**?

---

# Future Work

Possible extensions:

- global large-scale inference
- improved prompt strategies
- model fine-tuning for roof structures
- integration with GIS quality frameworks