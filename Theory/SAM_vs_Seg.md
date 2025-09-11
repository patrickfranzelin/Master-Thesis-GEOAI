# Building Segmentation Comparison – SegFormer vs SAM

This repository contains the output and preview results of two different approaches for automatic building segmentation based on high-resolution orthophotos. The goal is to extract vector polygons of buildings from raster imagery with minimal manual effort.

## Data Overview

- Input raster: `data/ortho_4.tif`  
  *(High-resolution orthophoto of a mountainous rural region)*

-  Outputs (stored in `outputs/`):
  - `buildings_segformer.gpkg`: Vectorized buildings predicted using the **SegFormer** model
  - `buildings_sam_tiles.gpkg`: Vectorized masks generated using **Segment Anything Model (SAM)** on tiled input
  - `overlay_preview_seg.png`: Visual comparison (SegFormer output overlayed on the orthophoto)
  - `overlay_preview.png`: Visual comparison (SAM output overlayed on the orthophoto)

---

## 1. SegFormer-Based Segmentation

**Approach**:  
SegFormer is a Vision Transformer-based semantic segmentation model. In this approach, a pre-trained SegFormer model was used to predict **semantic classes** (e.g., buildings vs. background). The building mask was then vectorized into polygons.

**Strengths**:
- Compact output focused on buildings
- Good generalization for rooftops in open terrain
- Relatively clean outlines

**Weaknesses**:
- May miss buildings with low contrast or shadows
- No detection of small annexes or partially occluded structures

**Preview**:  
![SegFormer Preview](outputs/overlay_preview_seg.png)

---

## 2. Segment Anything (SAM) – Tiled Inference

**Approach**:  
The SAM model (Segment Anything by Meta AI) was used in **zero-shot** mode on tiled patches of the raster. Each tile was segmented independently, and all resulting masks were filtered, merged, and vectorized.

**Strengths**:
- High geometric detail
- Captures even small structures and varied shapes
- Can be used for further refinement or as a pre-labeling tool

**Weaknesses**:
- No semantic understanding (e.g., it does not know what's a building)
- Over-segmentation: often detects trees, roads, textures, or noise
- Grid artifacts from tiling (visible in the overlay)

**Preview**:  
![SAM Preview](outputs/overlay_preview.png)

---

## Next Steps

- Quantitative evaluation using intersection-over-union (IoU) and comparison to reference data
- Possible fine-tuning of SegFormer with domain-specific building samples
- Use SAM as a refinement tool (e.g., edge refinement or post-filtering of SegFormer output)
---

## 4. Semantic Interpretation with Local LLMs (Optional Extension)

**Approach**:
In addition to geometric segmentation, it is possible to extract **semantic and functional information** from the segmented objects (e.g., types of buildings or roads) using **open-source Large Language Models (LLMs)** or **Multimodal Language Models (MLLMs)**.

This step enables contextual labeling such as:

* Classifying buildings (e.g., *school*, *hospital*, *warehouse*)
* Identifying road types (e.g., *local road*, *highway A1*, *unpaved path*)
* Detecting special features (e.g., *bridge section*, *courtyard*, *collapsed structure*)

This works by either describing the polygonal features in text form or directly supplying the image + polygon overlay to a multimodal model.

**Recommended Models**:

| Model Type           | Examples                        | Use Case                                                       |
| -------------------- | ------------------------------- | -------------------------------------------------------------- |
| **LLMs**             | LLaMA-2, Mistral, Falcon, Gemma | Text-based interpretation (e.g., classify a described polygon) |
| **Multimodal LLMs**  | LLaVA, Qwen-VL, Gemma-VLM       | Input image + prompt → output with contextual understanding    |
| **Hybrid Pipelines** | LLM + rules                     | Combine reasoning with domain knowledge                        |


## Summary

* Segmentation models deliver vector shapes, but adding **semantic meaning** to these shapes opens up advanced geospatial applications.
* Open-source LLMs and MLLMs provide an accessible, low-cost path to interpret and enrich raster-derived features.
* This is a promising direction for future development or an extended Master thesis module.

