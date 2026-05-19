# evaluate_buildings_arcgis.py
# ------------------------------------------------------------
# Compare regularized buildings and Microsoft buildings against
# TLM buildings as ground truth.
#
# Candidate 1: Regularized buildings
# Candidate 2: Microsoft buildings
# Ground truth: TLM buildings
#
# Important for interrupted regularization runs:
#   If the regularization process was interrupted, the regularized layer
#   does not cover the full AOI. Therefore, this script restricts the
#   evaluation to the Microsoft buildings for which a corresponding
#   regularized output is available.
#
# Evaluation logic:
#   1. Prepare AOI, regularized buildings, Microsoft buildings and TLM buildings.
#   2. Select available regularized buildings inside AOI.
#   3. Match Microsoft buildings to available regularized buildings.
#   4. Match TLM buildings to the processed Microsoft subset.
#   5. Evaluate:
#        - regularized buildings vs TLM subset
#        - Microsoft processed subset vs TLM subset
#
# Matching metric:
#   IoU = intersection_area / union_area
# ------------------------------------------------------------

import arcpy
import csv
import os
from pathlib import Path


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

PREDICTED_FC = (
    r"C:\Users\franz\Documents\ArcGIS\Projects\Test_Code_Switzerland"
    r"\Test_Code_Switzerland.gdb\detected_house_RegularizeBui"
)

MICROSOFT_FC = (
    r"C:\Users\franz\Documents\ArcGIS\Projects\Test_Code_Switzerland"
    r"\Test_Code_Switzerland.gdb\mainbuildings_Project_clip"
)

TLM_FC = (
    r"C:\Users\franz\Documents\ArcGIS\Projects\Test_Code_Switzerland"
    r"\Test_Code_Switzerland.gdb\Gebaeudeeinheit_Clip"
)

AOI_FC = (
    r"C:\git\Master-Thesis-GEOAI\data\switzerland_buildings"
    r"\aoi_switzerland.gdb\aoi_switzerland"
)

OUT_DIR = r"C:\git\Master-Thesis-GEOAI\data\evaluation_switzerland"

TARGET_EPSG = 2056  # CH1903+ / LV95

IOU_THRESHOLDS = [0.10, 0.25, 0.50]


# ------------------------------------------------------------
# EVALUATION MODE
# ------------------------------------------------------------

# True:
#   Evaluation is restricted to buildings for which a regularized output
#   is available. This is the correct mode if your regularization run was
#   interrupted and does not contain all buildings in the AOI.
#
# False:
#   Uses the full AOI.
MATCH_ONLY_AVAILABLE_REGULARIZED_BUILDINGS = True

# Used to match original Microsoft buildings to available regularized outputs.
# A small buffer is useful because regularization can slightly move/shrink/grow polygons.
REGULARIZED_TO_MICROSOFT_MATCH_BUFFER_METERS = 1

# Used to select corresponding TLM buildings around the processed Microsoft subset.
MICROSOFT_TO_TLM_MATCH_BUFFER_METERS = 1

# STATUS = 0 means successfully regularized.
# STATUS = 1 usually means copied original / not successfully regularized.
#
# For the actual regularized candidate evaluation, you probably want only STATUS = 0.
USE_ONLY_STATUS0_AS_REGULARIZED_CANDIDATE = True

# For reconstructing the processed area, it is usually better to use ALL available
# regularized outputs inside the AOI, not only STATUS = 0.
#
# Reason:
# STATUS = 1 may still indicate that the building was attempted by ArcGIS but copied.
# Therefore, it belongs to the processed subset.
USE_ALL_AVAILABLE_REGULARIZED_FOR_SOURCE_MATCH = True


# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------

arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

scratch_gdb = str(out_dir / "evaluation_scratch.gdb")

if not arcpy.Exists(scratch_gdb):
    arcpy.management.CreateFileGDB(str(out_dir), "evaluation_scratch.gdb")

arcpy.env.workspace = scratch_gdb

target_sr = arcpy.SpatialReference(TARGET_EPSG)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def msg(text):
    print(text)


def safe_name(name):
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def count_features(fc):
    return int(arcpy.management.GetCount(fc)[0])


def delete_if_exists(path):
    if arcpy.Exists(path):
        arcpy.management.Delete(path)


def delete_field_if_exists(fc, field_name):
    existing = [f.name.upper() for f in arcpy.ListFields(fc)]
    if field_name.upper() in existing:
        arcpy.management.DeleteField(fc, field_name)


def add_area_field(fc, field_name="AREA_M2"):
    fields = [f.name for f in arcpy.ListFields(fc)]

    if field_name not in fields:
        arcpy.management.AddField(fc, field_name, "DOUBLE")

    arcpy.management.CalculateGeometryAttributes(
        fc,
        [[field_name, "AREA"]],
        area_unit="SQUARE_METERS",
    )


def add_fresh_id_field(fc, id_field):
    """
    Adds a fresh integer ID field based on ObjectID.
    Existing field with the same name is deleted first.

    This is important because the same layer may be used once as candidate
    and once as reference. Old ID fields can otherwise cause wrong joins
    after PairwiseIntersect.
    """
    delete_field_if_exists(fc, id_field)

    oid_field = arcpy.Describe(fc).OIDFieldName

    arcpy.management.AddField(fc, id_field, "LONG")

    arcpy.management.CalculateField(
        fc,
        id_field,
        f"!{oid_field}!",
        "PYTHON3",
    )


def print_extent(fc, name):
    desc = arcpy.Describe(fc)
    ext = desc.extent
    sr = desc.spatialReference

    msg(f"\nExtent of {name}:")
    msg(
        f"  CRS: {sr.name if sr else 'Unknown'} / "
        f"EPSG: {sr.factoryCode if sr else 'Unknown'}"
    )
    msg(f"  XMin: {ext.XMin}")
    msg(f"  YMin: {ext.YMin}")
    msg(f"  XMax: {ext.XMax}")
    msg(f"  YMax: {ext.YMax}")


def copy_project_clean(input_fc, output_name, target_spatial_reference):
    """
    Copies or projects input feature class into scratch GDB.
    """
    output_fc = os.path.join(scratch_gdb, output_name)
    delete_if_exists(output_fc)

    desc = arcpy.Describe(input_fc)
    src_sr = desc.spatialReference

    msg(f"\nPreparing: {output_name}")
    msg(f"  Source: {input_fc}")
    msg(f"  Source CRS: {src_sr.name if src_sr else 'Unknown'}")

    temp_fc = os.path.join(scratch_gdb, f"{output_name}_raw")
    delete_if_exists(temp_fc)

    if src_sr is None or src_sr.name == "Unknown":
        raise ValueError(f"Input has unknown CRS: {input_fc}")

    if src_sr.factoryCode == target_spatial_reference.factoryCode:
        arcpy.management.CopyFeatures(input_fc, temp_fc)
    else:
        msg(f"  Projecting to {target_spatial_reference.name}")
        arcpy.management.Project(
            in_dataset=input_fc,
            out_dataset=temp_fc,
            out_coor_system=target_spatial_reference,
        )

    arcpy.management.RepairGeometry(temp_fc, "DELETE_NULL")
    arcpy.management.CopyFeatures(temp_fc, output_fc)

    add_area_field(output_fc, "AREA_M2")

    msg(f"  Output: {output_fc}")
    msg(f"  Feature count: {count_features(output_fc)}")

    return output_fc


def select_features_by_aoi(input_fc, aoi_fc, output_name):
    """
    Selects only features intersecting the AOI.
    """
    output_fc = os.path.join(scratch_gdb, output_name)
    delete_if_exists(output_fc)

    msg(f"\nSelecting {output_name} inside AOI")

    lyr = f"{safe_name(output_name)}_lyr"
    delete_if_exists(lyr)

    arcpy.management.MakeFeatureLayer(input_fc, lyr)

    arcpy.management.SelectLayerByLocation(
        in_layer=lyr,
        overlap_type="INTERSECT",
        select_features=aoi_fc,
        selection_type="NEW_SELECTION",
    )

    arcpy.management.CopyFeatures(lyr, output_fc)
    arcpy.management.RepairGeometry(output_fc, "DELETE_NULL")
    add_area_field(output_fc, "AREA_M2")

    msg(f"  Selected features: {count_features(output_fc)}")
    msg(f"  Output: {output_fc}")

    return output_fc


def filter_regularized_status0(predicted_fc, output_name):
    """
    ArcGIS Regularize Building Footprint:
    STATUS = 0 means successfully regularized.
    STATUS = 1 means original copied / not successfully regularized.

    If no STATUS field exists, all features are used.
    """
    fields = [f.name.upper() for f in arcpy.ListFields(predicted_fc)]

    output_fc = os.path.join(scratch_gdb, output_name)
    delete_if_exists(output_fc)

    if "STATUS" in fields:
        msg("\nFiltering regularized buildings to STATUS = 0")
        arcpy.analysis.Select(
            predicted_fc,
            output_fc,
            "STATUS = 0",
        )
        msg("  Using only successfully regularized buildings.")
    else:
        msg("\nNo STATUS field found.")
        msg("  Using all available regularized buildings inside AOI.")
        arcpy.management.CopyFeatures(predicted_fc, output_fc)

    arcpy.management.RepairGeometry(output_fc, "DELETE_NULL")
    add_area_field(output_fc, "AREA_M2")

    msg(f"  Regularized candidate count: {count_features(output_fc)}")

    return output_fc


def buffer_features(input_fc, output_name, buffer_m):
    """
    Creates a dissolved buffer around features.
    """
    output_fc = os.path.join(scratch_gdb, output_name)
    delete_if_exists(output_fc)

    arcpy.analysis.Buffer(
        in_features=input_fc,
        out_feature_class=output_fc,
        buffer_distance_or_field=f"{buffer_m} Meters",
        dissolve_option="ALL",
    )

    arcpy.management.RepairGeometry(output_fc, "DELETE_NULL")

    return output_fc


def select_by_location_to_output(
    input_fc,
    selector_fc,
    output_name,
    overlap_type="INTERSECT",
):
    """
    Selects features from input_fc by spatial relation to selector_fc.
    """
    output_fc = os.path.join(scratch_gdb, output_name)
    delete_if_exists(output_fc)

    lyr = f"{safe_name(output_name)}_lyr"
    delete_if_exists(lyr)

    arcpy.management.MakeFeatureLayer(input_fc, lyr)

    arcpy.management.SelectLayerByLocation(
        in_layer=lyr,
        overlap_type=overlap_type,
        select_features=selector_fc,
        selection_type="NEW_SELECTION",
    )

    arcpy.management.CopyFeatures(lyr, output_fc)
    arcpy.management.RepairGeometry(output_fc, "DELETE_NULL")
    add_area_field(output_fc, "AREA_M2")

    return output_fc


def select_microsoft_matching_available_regularized(
    microsoft_fc,
    available_regularized_fc,
    output_name,
    buffer_m=3,
):
    """
    Selects Microsoft buildings that correspond to available regularized outputs.

    This reconstructs the original Microsoft source subset that was actually
    processed before the regularization run was interrupted.
    """
    msg(f"\nSelecting Microsoft buildings matching available regularized outputs")
    msg(f"  Match buffer distance: {buffer_m} m")

    buffer_fc = buffer_features(
        input_fc=available_regularized_fc,
        output_name=f"{output_name}_regularized_buffer",
        buffer_m=buffer_m,
    )

    output_fc = select_by_location_to_output(
        input_fc=microsoft_fc,
        selector_fc=buffer_fc,
        output_name=output_name,
        overlap_type="INTERSECT",
    )

    msg(f"  Selected Microsoft source buildings: {count_features(output_fc)}")
    msg(f"  Output: {output_fc}")

    return output_fc


def select_tlm_matching_processed_microsoft(
    tlm_fc,
    processed_microsoft_fc,
    output_name,
    buffer_m=3,
):
    """
    Selects TLM buildings that correspond to the processed Microsoft subset.

    This avoids evaluating against TLM buildings that were outside the part
    of the AOI where regularized outputs are available.
    """
    msg(f"\nSelecting TLM buildings matching processed Microsoft subset")
    msg(f"  Match buffer distance: {buffer_m} m")

    buffer_fc = buffer_features(
        input_fc=processed_microsoft_fc,
        output_name=f"{output_name}_microsoft_buffer",
        buffer_m=buffer_m,
    )

    output_fc = select_by_location_to_output(
        input_fc=tlm_fc,
        selector_fc=buffer_fc,
        output_name=output_name,
        overlap_type="INTERSECT",
    )

    msg(f"  Selected TLM ground-truth buildings: {count_features(output_fc)}")
    msg(f"  Output: {output_fc}")

    return output_fc


def select_regularized_matching_processed_microsoft(
    regularized_fc,
    processed_microsoft_fc,
    output_name,
    buffer_m=3,
):
    """
    Restricts the regularized candidate to the same processed Microsoft subset.

    Usually this should not remove many features, but it makes sure the
    evaluated regularized layer, Microsoft layer and TLM layer all describe
    the same processed building universe.
    """
    msg(f"\nRestricting regularized buildings to processed Microsoft subset")
    msg(f"  Match buffer distance: {buffer_m} m")

    buffer_fc = buffer_features(
        input_fc=processed_microsoft_fc,
        output_name=f"{output_name}_microsoft_buffer",
        buffer_m=buffer_m,
    )

    output_fc = select_by_location_to_output(
        input_fc=regularized_fc,
        selector_fc=buffer_fc,
        output_name=output_name,
        overlap_type="INTERSECT",
    )

    msg(f"  Selected regularized candidate buildings: {count_features(output_fc)}")
    msg(f"  Output: {output_fc}")

    return output_fc


def build_intersection_pairs(candidate_fc, gt_fc, label):
    """
    Creates candidate-ground-truth intersection pairs.

    candidate_fc:
        Regularized buildings or Microsoft buildings.

    gt_fc:
        TLM buildings.

    Returns:
        candidate_id, gt_id, candidate_area, gt_area, intersection_area,
        union_area, IoU.
    """
    msg("\n" + "-" * 70)
    msg(f"Building intersection pairs for: {label}")
    msg("-" * 70)

    cand_id = "CAND_ID"
    gt_id = "GT_ID"

    # Fresh fields avoid ID collisions between comparisons.
    add_fresh_id_field(candidate_fc, cand_id)
    add_fresh_id_field(gt_fc, gt_id)

    inter_fc = os.path.join(scratch_gdb, f"intersections_{safe_name(label)}")
    delete_if_exists(inter_fc)

    arcpy.analysis.PairwiseIntersect(
        in_features=[candidate_fc, gt_fc],
        out_feature_class=inter_fc,
        join_attributes="ALL",
    )

    add_area_field(inter_fc, "INTER_AREA")

    fields = [f.name.upper() for f in arcpy.ListFields(inter_fc)]

    if cand_id.upper() not in fields:
        raise RuntimeError(f"Missing candidate ID field after intersection: {cand_id}")

    if gt_id.upper() not in fields:
        raise RuntimeError(f"Missing ground-truth ID field after intersection: {gt_id}")

    candidate_area_by_id = {}
    with arcpy.da.SearchCursor(candidate_fc, [cand_id, "AREA_M2"]) as cur:
        for cid, area in cur:
            candidate_area_by_id[cid] = area

    gt_area_by_id = {}
    with arcpy.da.SearchCursor(gt_fc, [gt_id, "AREA_M2"]) as cur:
        for gid, area in cur:
            gt_area_by_id[gid] = area

    pairs = []
    invalid_iou_count = 0

    with arcpy.da.SearchCursor(inter_fc, [cand_id, gt_id, "INTER_AREA"]) as cur:
        for cid, gid, inter_area in cur:
            if cid is None or gid is None:
                continue

            candidate_area = candidate_area_by_id.get(cid)
            gt_area = gt_area_by_id.get(gid)

            if not candidate_area or not gt_area or not inter_area:
                continue

            union_area = candidate_area + gt_area - inter_area

            if union_area <= 0:
                continue

            iou = inter_area / union_area

            # IoU must never be larger than 1.
            # If this happens, something is wrong with ID fields or geometry areas.
            if iou > 1.000001:
                invalid_iou_count += 1
                continue

            pairs.append(
                {
                    "candidate_id": cid,
                    "gt_id": gid,
                    "candidate_area_m2": candidate_area,
                    "gt_area_m2": gt_area,
                    "intersection_area_m2": inter_area,
                    "union_area_m2": union_area,
                    "iou": iou,
                }
            )

    msg(f"  Candidate intersecting pairs: {len(pairs)}")

    if invalid_iou_count > 0:
        msg(f"  WARNING: skipped {invalid_iou_count} invalid IoU values > 1")

    return pairs


def greedy_match_by_iou(pairs, threshold):
    """
    One-to-one greedy matching by descending IoU.
    """
    pairs_sorted = sorted(pairs, key=lambda x: x["iou"], reverse=True)

    matched_candidates = set()
    matched_gt = set()
    matches = []

    for pair in pairs_sorted:
        if pair["iou"] < threshold:
            continue

        cid = pair["candidate_id"]
        gid = pair["gt_id"]

        if cid in matched_candidates or gid in matched_gt:
            continue

        matched_candidates.add(cid)
        matched_gt.add(gid)
        matches.append(pair)

    return matches, matched_candidates, matched_gt


def export_pairs_csv(pairs, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "gt_id",
                "candidate_area_m2",
                "gt_area_m2",
                "intersection_area_m2",
                "union_area_m2",
                "iou",
            ],
        )
        writer.writeheader()

        for row in pairs:
            writer.writerow(row)


def chunk_list(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def export_unmatched_features(fc, id_field, unmatched_ids, output_fc):
    if not unmatched_ids:
        msg(f"  No unmatched features for {output_fc}")
        return None

    delete_if_exists(output_fc)

    lyr = f"unmatched_{safe_name(Path(output_fc).name)}_lyr"
    delete_if_exists(lyr)

    arcpy.management.MakeFeatureLayer(fc, lyr)

    temp_outputs = []

    for i, chunk in enumerate(chunk_list(list(unmatched_ids), 900)):
        values = ",".join(str(v) for v in chunk)
        where = f"{arcpy.AddFieldDelimiters(fc, id_field)} IN ({values})"

        arcpy.management.SelectLayerByAttribute(
            lyr,
            "NEW_SELECTION",
            where,
        )

        temp_fc = os.path.join(scratch_gdb, f"{Path(output_fc).name}_part_{i}")
        delete_if_exists(temp_fc)

        arcpy.management.CopyFeatures(lyr, temp_fc)
        temp_outputs.append(temp_fc)

    if len(temp_outputs) == 1:
        arcpy.management.CopyFeatures(temp_outputs[0], output_fc)
    else:
        arcpy.management.Merge(temp_outputs, output_fc)

    msg(f"  Exported unmatched features: {output_fc}")

    return output_fc


def evaluate_candidate_against_tlm(candidate_fc, tlm_fc, label):
    """
    Evaluates one candidate dataset against TLM ground truth.
    """
    msg("\n" + "=" * 70)
    msg(f"Evaluating candidate against TLM: {label}")
    msg("=" * 70)

    pairs = build_intersection_pairs(candidate_fc, tlm_fc, label)

    candidate_count = count_features(candidate_fc)
    gt_count = count_features(tlm_fc)

    all_pairs_csv = out_dir / f"all_intersection_pairs_{safe_name(label)}.csv"
    export_pairs_csv(pairs, all_pairs_csv)

    summary_rows = []

    for threshold in IOU_THRESHOLDS:
        matches, matched_candidates, matched_gt = greedy_match_by_iou(
            pairs,
            threshold,
        )

        tp = len(matches)
        fp = candidate_count - len(matched_candidates)
        fn = gt_count - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        mean_iou = (
            sum(m["iou"] for m in matches) / len(matches)
            if matches
            else 0
        )

        row = {
            "comparison": label,
            "ground_truth": "TLM",
            "iou_threshold": threshold,
            "candidate_count": candidate_count,
            "tlm_count": gt_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_matched": mean_iou,
        }

        summary_rows.append(row)

        matches_csv = out_dir / f"matches_{safe_name(label)}_iou_{threshold}.csv"
        export_pairs_csv(matches, matches_csv)

        msg(f"\nIoU threshold: {threshold}")
        msg(f"  Candidate count: {candidate_count}")
        msg(f"  TLM count:       {gt_count}")
        msg(f"  TP: {tp}")
        msg(f"  FP: {fp}")
        msg(f"  FN: {fn}")
        msg(f"  Precision: {precision:.3f}")
        msg(f"  Recall:    {recall:.3f}")
        msg(f"  F1:        {f1:.3f}")
        msg(f"  Mean IoU:  {mean_iou:.3f}")

        if mean_iou > 1:
            raise RuntimeError(
                f"Invalid mean IoU > 1 detected for {label}. "
                "Check ID fields and geometry areas."
            )

        if threshold == 0.50:
            all_candidate_ids = set()
            with arcpy.da.SearchCursor(candidate_fc, ["CAND_ID"]) as cur:
                for (cid,) in cur:
                    all_candidate_ids.add(cid)

            all_gt_ids = set()
            with arcpy.da.SearchCursor(tlm_fc, ["GT_ID"]) as cur:
                for (gid,) in cur:
                    all_gt_ids.add(gid)

            false_positive_ids = all_candidate_ids - matched_candidates
            false_negative_ids = all_gt_ids - matched_gt

            fp_fc = os.path.join(
                scratch_gdb,
                f"false_positives_{safe_name(label)}_iou50",
            )

            fn_fc = os.path.join(
                scratch_gdb,
                f"false_negatives_{safe_name(label)}_iou50",
            )

            export_unmatched_features(
                candidate_fc,
                "CAND_ID",
                false_positive_ids,
                fp_fc,
            )

            export_unmatched_features(
                tlm_fc,
                "GT_ID",
                false_negative_ids,
                fn_fc,
            )

    return summary_rows


def export_summary(summary_rows):
    summary_csv = out_dir / "evaluation_summary.csv"

    fieldnames = [
        "comparison",
        "ground_truth",
        "iou_threshold",
        "candidate_count",
        "tlm_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_iou_matched",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    msg("\nSummary exported to:")
    msg(str(summary_csv))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    msg("Starting building footprint evaluation against TLM ground truth...")

    if MATCH_ONLY_AVAILABLE_REGULARIZED_BUILDINGS:
        msg("\nMATCH_ONLY_AVAILABLE_REGULARIZED_BUILDINGS = True")
        msg(
            "Evaluation will be restricted to Microsoft buildings for which "
            "a corresponding regularized output is available."
        )
        msg(
            "This avoids counting buildings as false negatives only because "
            "the regularization run was interrupted."
        )
    else:
        msg("\nMATCH_ONLY_AVAILABLE_REGULARIZED_BUILDINGS = False")
        msg("Full AOI will be used.")

    # ------------------------------------------------------------
    # 1. Prepare AOI
    # ------------------------------------------------------------
    aoi = copy_project_clean(
        AOI_FC,
        "aoi_switzerland_lv95",
        target_sr,
    )

    # ------------------------------------------------------------
    # 2. Prepare regularized buildings
    # ------------------------------------------------------------
    regularized_all = copy_project_clean(
        PREDICTED_FC,
        "regularized_buildings_lv95_all",
        target_sr,
    )

    regularized_aoi = select_features_by_aoi(
        regularized_all,
        aoi,
        "regularized_buildings_lv95_aoi",
    )

    if USE_ONLY_STATUS0_AS_REGULARIZED_CANDIDATE:
        regularized_candidate_raw = filter_regularized_status0(
            regularized_aoi,
            "regularized_buildings_lv95_aoi_status0",
        )
    else:
        msg("\nUsing all available regularized buildings as regularized candidate.")
        regularized_candidate_raw = os.path.join(
            scratch_gdb,
            "regularized_buildings_lv95_aoi_all_available_candidate",
        )
        delete_if_exists(regularized_candidate_raw)
        arcpy.management.CopyFeatures(regularized_aoi, regularized_candidate_raw)
        arcpy.management.RepairGeometry(regularized_candidate_raw, "DELETE_NULL")
        add_area_field(regularized_candidate_raw, "AREA_M2")

    # ------------------------------------------------------------
    # 3. Prepare Microsoft buildings
    # ------------------------------------------------------------
    microsoft_all = copy_project_clean(
        MICROSOFT_FC,
        "microsoft_buildings_lv95_all",
        target_sr,
    )

    microsoft_aoi = select_features_by_aoi(
        microsoft_all,
        aoi,
        "microsoft_buildings_lv95_aoi",
    )

    # ------------------------------------------------------------
    # 4. Prepare TLM ground truth
    # ------------------------------------------------------------
    tlm_all = copy_project_clean(
        TLM_FC,
        "tlm_buildings_lv95_all",
        target_sr,
    )

    tlm_aoi = select_features_by_aoi(
        tlm_all,
        aoi,
        "tlm_buildings_lv95_aoi",
    )

    # ------------------------------------------------------------
    # 5. Restrict evaluation to available processed buildings
    # ------------------------------------------------------------
    if MATCH_ONLY_AVAILABLE_REGULARIZED_BUILDINGS:
        if USE_ALL_AVAILABLE_REGULARIZED_FOR_SOURCE_MATCH:
            regularized_source_match_layer = regularized_aoi
            msg(
                "\nUsing all available regularized outputs to reconstruct "
                "the processed Microsoft source subset."
            )
        else:
            regularized_source_match_layer = regularized_candidate_raw
            msg(
                "\nUsing only the regularized candidate layer to reconstruct "
                "the processed Microsoft source subset."
            )

        microsoft_candidate = select_microsoft_matching_available_regularized(
            microsoft_fc=microsoft_aoi,
            available_regularized_fc=regularized_source_match_layer,
            output_name="microsoft_buildings_lv95_matching_available_regularized",
            buffer_m=REGULARIZED_TO_MICROSOFT_MATCH_BUFFER_METERS,
        )

        regularized_candidate = select_regularized_matching_processed_microsoft(
            regularized_fc=regularized_candidate_raw,
            processed_microsoft_fc=microsoft_candidate,
            output_name="regularized_buildings_lv95_matching_processed_microsoft",
            buffer_m=REGULARIZED_TO_MICROSOFT_MATCH_BUFFER_METERS,
        )

        tlm_ground_truth = select_tlm_matching_processed_microsoft(
            tlm_fc=tlm_aoi,
            processed_microsoft_fc=microsoft_candidate,
            output_name="tlm_buildings_lv95_matching_processed_microsoft",
            buffer_m=MICROSOFT_TO_TLM_MATCH_BUFFER_METERS,
        )

        regularized_label = "regularized_vs_tlm_matching_available_regularized"
        microsoft_label = "microsoft_vs_tlm_matching_available_regularized"

    else:
        regularized_candidate = regularized_candidate_raw
        microsoft_candidate = microsoft_aoi
        tlm_ground_truth = tlm_aoi

        regularized_label = "regularized_vs_tlm_aoi"
        microsoft_label = "microsoft_vs_tlm_aoi"

    add_area_field(regularized_candidate, "AREA_M2")
    add_area_field(microsoft_candidate, "AREA_M2")
    add_area_field(tlm_ground_truth, "AREA_M2")

    # ------------------------------------------------------------
    # 6. Debug extents and counts
    # ------------------------------------------------------------
    print_extent(aoi, "AOI")
    print_extent(regularized_candidate, "regularized_candidate")
    print_extent(microsoft_candidate, "microsoft_candidate")
    print_extent(tlm_ground_truth, "tlm_ground_truth")

    regularized_count = count_features(regularized_candidate)
    microsoft_count = count_features(microsoft_candidate)
    tlm_count = count_features(tlm_ground_truth)

    msg("\nCounts used for evaluation:")
    msg(f"  Regularized candidate buildings: {regularized_count}")
    msg(f"  Microsoft candidate buildings:   {microsoft_count}")
    msg(f"  TLM ground-truth buildings:       {tlm_count}")

    if regularized_count == 0:
        raise RuntimeError("No regularized buildings found.")

    if microsoft_count == 0:
        raise RuntimeError("No Microsoft buildings found.")

    if tlm_count == 0:
        raise RuntimeError("No TLM buildings found.")

    # ------------------------------------------------------------
    # 7. Evaluate both candidates against TLM
    # ------------------------------------------------------------
    all_summary_rows = []

    all_summary_rows.extend(
        evaluate_candidate_against_tlm(
            regularized_candidate,
            tlm_ground_truth,
            regularized_label,
        )
    )

    all_summary_rows.extend(
        evaluate_candidate_against_tlm(
            microsoft_candidate,
            tlm_ground_truth,
            microsoft_label,
        )
    )

    # ------------------------------------------------------------
    # 8. Export summary
    # ------------------------------------------------------------
    export_summary(all_summary_rows)

    msg("\nDone.")


if __name__ == "__main__":
    main()