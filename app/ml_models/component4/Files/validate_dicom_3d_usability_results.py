"""
validate_dicom_3d_usability_results.py

STANDALONE, READ-ONLY validation of the prior 3D usability assessment's
classification, particularly the MULTIPLE_ACQUISITIONS category (250
of 677 series - unusually high, needs re-verification against real
geometry rather than trusted as-is).

CRITICAL REFRAMING (Part 2 of the request): the prior assessment
script (assess_dicom_3d_usability.py) flags MULTIPLE_ACQUISITIONS
whenever EITHER of two supporting signals appears: >1 AcquisitionNumber/
AcquisitionTime value, OR >=2 long monotonic runs in InstanceNumber-
recorded order. Both are real signals, but NEITHER alone proves the
slices cannot form one coherent volume - e.g. two AcquisitionNumbers
covering two DIFFERENT, non-overlapping z-ranges (acquisition 1:
z=100..50, acquisition 2: z=49..0) still sort into ONE clean,
monotonic, non-overlapping 100..0 sequence with zero duplicate
positions - a spatially coherent series, even though it shows "2
acquisitions" and "2 InstanceNumber-order runs".

This script implements the PRECISE geometric test Part 2 actually
asks for: after ordering ALL slices by their true ImagePositionPatient
z-value (never by InstanceNumber for this specific test), check
whether any two slices occupy the SAME (or near-identical, within
tolerance) physical position. If yes, that is genuine, direct evidence
of real spatial conflict (the same location scanned more than once) -
confirmed multi-acquisition. If no such overlap exists - the
position-sorted sequence is strictly monotonic with no duplicates,
regardless of gaps (per the explicit "gaps do not cause rejection"
rule) - the series is SPATIALLY COHERENT, and the original
MULTIPLE_ACQUISITIONS label is flagged as a likely false positive,
even if AcquisitionNumber/AcquisitionTime/InstanceNumber-order
patterns still look unusual (reported as supporting context only, not
as the determining factor).

Does NOT change assess_dicom_3d_usability.py or its output files.
Reads them as input, produces an independent, separate validation
report with its own recommended_classification field per series -
this is advisory output for you to review, not an automatic
reclassification of anything.

READ-ONLY. Uses stop_before_pixels=True throughout. Never calls
shutil.copy/move, os.rename, os.remove, Path.unlink, or any equivalent
write/delete operation. All writes go only to --output.

Usage:
    python validate_dicom_3d_usability_results.py \
        --dataset-root "D:/DICOM/archive/dicom_3d_working_dataset" \
        --usability-csv "D:/DICOM/archive/dicom_3d_usability/3d_series_usability.csv" \
        --output "D:/DICOM/archive/dicom_3d_usability_validation"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pydicom

# --- Documented, reasoned thresholds - identical values to
# assess_dicom_3d_usability.py where the same concept applies, so
# results stay directly comparable. ---------------------------------

DUPLICATE_POSITION_TOLERANCE_MM = 0.01
STANDARD_AXIAL_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
AXIAL_TOLERANCE = 0.05
MIN_SWEEP_LENGTH = 3

# PixelSpacing values within this fraction of each other are treated
# as rounding noise rather than a real change - stated, reasoned
# choice, not empirically derived from this project's own data.
PIXEL_SPACING_NOISE_TOLERANCE = 0.02  # 2%

FULL_TAG_LIST = [
    "SeriesNumber", "AcquisitionNumber", "AcquisitionDate", "AcquisitionTime",
    "ContentTime", "SliceThickness", "SpacingBetweenSlices",
]


def is_standard_axial(iop_values, tolerance: float = AXIAL_TOLERANCE) -> bool:
    try:
        iop = np.array([float(v) for v in iop_values])
    except (TypeError, ValueError):
        return False
    if iop.shape != (6,):
        return False
    return bool(np.allclose(iop, STANDARD_AXIAL_IOP, atol=tolerance))


def count_monotonic_runs(z: list[float]) -> list[int]:
    if len(z) < 2:
        return [len(z)]
    diffs = np.diff(z)
    signs = np.sign(diffs)
    runs = []
    cur_len = 1
    cur_sign = signs[0] if signs[0] != 0 else 1
    for s in signs[1:]:
        if s == cur_sign or s == 0:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_len = 1
            cur_sign = s
    runs.append(cur_len)
    return runs


# ---------------------------------------------------------------------
# Step 1: build a metadata index of the working dataset - ONE pass.
# ---------------------------------------------------------------------

def build_index(dataset_root: str) -> dict:
    print(f"Building metadata index of {dataset_root} (one pass, "
          f"metadata only, stop_before_pixels=True)...")
    index = defaultdict(list)
    total = 0
    unreadable = 0

    class_folders = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    for folder in class_folders:
        folder_path = os.path.join(dataset_root, folder)
        fnames = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
        for fname in fnames:
            total += 1
            if total % 2000 == 0:
                print(f"  ...indexed {total} files so far")
            fpath = os.path.join(folder_path, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception:
                unreadable += 1
                continue

            patient_id = getattr(ds, "PatientID", None)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            series_uid = getattr(ds, "SeriesInstanceUID", None)

            meta = {
                "path": fpath,
                "modality": getattr(ds, "Modality", None),
                "instance_number": int(getattr(ds, "InstanceNumber")) if hasattr(ds, "InstanceNumber") else None,
                "image_position": [float(v) for v in ds.ImagePositionPatient] if hasattr(ds, "ImagePositionPatient") else None,
                "image_orientation": [float(v) for v in ds.ImageOrientationPatient] if hasattr(ds, "ImageOrientationPatient") else None,
                "pixel_spacing": [float(v) for v in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else None,
                "rows": int(ds.Rows) if hasattr(ds, "Rows") else None,
                "columns": int(ds.Columns) if hasattr(ds, "Columns") else None,
                "has_patient_id": bool(patient_id),
                "has_study_uid": bool(study_uid),
                "has_series_uid": bool(series_uid),
            }
            for tag in FULL_TAG_LIST:
                meta[tag] = str(getattr(ds, tag, "")) or None

            if patient_id and study_uid and series_uid:
                key = (str(patient_id), str(study_uid), str(series_uid))
                index[key].append(meta)
            else:
                # Still record under a per-file pseudo-key so OTHER-category
                # investigation can find files even when grouping keys are
                # incomplete - the ORIGINAL assessment would have excluded
                # these from any series group entirely.
                index[("__unkeyed__", fpath, "")] = [meta]

    print(f"Indexed {total} files ({unreadable} unreadable). "
          f"{len([k for k in index if k[0] != '__unkeyed__'])} distinct series found.")
    return dict(index)


# ---------------------------------------------------------------------
# Core geometric analysis, shared by multiple-acquisition and usable
# control-group analysis.
# ---------------------------------------------------------------------

def geometric_analysis(files: list[dict]) -> dict:
    """The precise, position-based analysis described in the module
    docstring. Returns a dict of all Part 1 metrics plus the revised
    spatially_coherent determination.
    """
    n = len(files)
    r = {"number_of_slices": n}

    with_pos = [f for f in files if f["image_position"] is not None]
    r["unique_acquisition_numbers"] = len(set(f["AcquisitionNumber"] for f in files if f["AcquisitionNumber"]))
    r["unique_acquisition_times"] = len(set(f["AcquisitionTime"] for f in files if f["AcquisitionTime"]))
    r["unique_series_numbers"] = len(set(f["SeriesNumber"] for f in files if f["SeriesNumber"]))
    r["unique_orientations"] = len(set(tuple(f["image_orientation"]) for f in files if f["image_orientation"]))
    r["unique_pixel_spacings"] = len(set(tuple(f["pixel_spacing"]) for f in files if f["pixel_spacing"]))
    r["unique_dimensions"] = len(set((f["rows"], f["columns"]) for f in files))

    if len(with_pos) < 2:
        r.update({
            "unique_z_positions": len(with_pos), "duplicate_z_positions": 0,
            "min_z": None, "max_z": None, "median_spacing": None,
            "detected_sweep_count": None, "sweep_lengths": None,
            "spatially_coherent": None,
            "spatial_evidence": "Fewer than 2 slices with ImagePositionPatient - cannot assess.",
        })
        return r

    # --- THE KEY TEST: position-sorted order, checked for genuine
    # spatial overlap/duplication - this is what actually determines
    # spatially_coherent, NOT AcquisitionNumber/Time/InstanceNumber
    # patterns, which are recorded separately as supporting context.
    position_ordered = sorted(with_pos, key=lambda f: f["image_position"][2])
    z_by_position = np.array([f["image_position"][2] for f in position_ordered])
    pos_diffs = np.abs(np.diff(z_by_position))
    duplicate_count = int((pos_diffs < DUPLICATE_POSITION_TOLERANCE_MM).sum())
    unique_z = int(len(z_by_position) - duplicate_count)

    r["unique_z_positions"] = unique_z
    r["duplicate_z_positions"] = duplicate_count
    r["min_z"] = float(z_by_position.min())
    r["max_z"] = float(z_by_position.max())
    nonzero_diffs = pos_diffs[pos_diffs >= DUPLICATE_POSITION_TOLERANCE_MM]
    r["median_spacing"] = float(np.median(nonzero_diffs)) if len(nonzero_diffs) else 0.0

    # Supporting context only - InstanceNumber-order run detection,
    # reported but NOT used to determine spatially_coherent.
    with_instance = [f for f in files if f["instance_number"] is not None and f["image_position"] is not None]
    if len(with_instance) >= 2:
        instance_ordered = sorted(with_instance, key=lambda f: f["instance_number"])
        z_by_instance = [f["image_position"][2] for f in instance_ordered]
        run_lengths = count_monotonic_runs(z_by_instance)
        long_runs = [x for x in run_lengths if x >= MIN_SWEEP_LENGTH]
    else:
        long_runs = []
    r["detected_sweep_count"] = len(long_runs)
    r["sweep_lengths"] = long_runs

    # --- spatially_coherent: the ONLY determining test is genuine
    # position overlap. Zero duplicate positions => coherent, REGARDLESS
    # of AcquisitionNumber/Time/InstanceNumber-run patterns.
    r["spatially_coherent"] = duplicate_count == 0

    if duplicate_count > 0:
        r["spatial_evidence"] = (
            f"{duplicate_count} pair(s) of slices occupy the same physical "
            f"z-position (within {DUPLICATE_POSITION_TOLERANCE_MM}mm) - direct "
            f"evidence of real spatial conflict/redundant acquisition."
        )
    else:
        r["spatial_evidence"] = (
            "Zero duplicate/overlapping physical positions after true "
            "position-based ordering - slices form one spatially coherent, "
            "non-overlapping sequence (gaps, if any, do not affect this)."
        )

    return r


def classify_multiple_acquisition_series(geo: dict) -> tuple[str, str]:
    """Returns (recommended_classification, evidence_reason)."""
    if geo.get("spatially_coherent") is None:
        return "INSUFFICIENT_EVIDENCE", geo.get("spatial_evidence", "")
    if geo["spatially_coherent"]:
        extra = []
        if geo["unique_acquisition_numbers"] > 1:
            extra.append(f"{geo['unique_acquisition_numbers']} AcquisitionNumbers")
        if geo["detected_sweep_count"] and geo["detected_sweep_count"] >= 2:
            extra.append(f"{geo['detected_sweep_count']} InstanceNumber-order runs")
        context = f" (supporting signals present but not determinative: {', '.join(extra)})" if extra else ""
        return "LIKELY_USABLE_FALSE_POSITIVE", geo["spatial_evidence"] + context
    else:
        return "CONFIRMED_MULTIPLE_ACQUISITIONS", geo["spatial_evidence"]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--usability-csv", required=True,
                         help="Path to the EXISTING 3d_series_usability.csv "
                              "produced by assess_dicom_3d_usability.py.")
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_usability_validation")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.usability_csv):
        print(f"ERROR: usability CSV not found: {args.usability_csv}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("DICOM 3D USABILITY RESULTS VALIDATION - READ-ONLY")
    print("=" * 70)

    usability_df = pd.read_csv(args.usability_csv)
    index = build_index(args.dataset_root)

    def get_files(row):
        key = (str(row["patient_id"]), str(row["study_uid"]), str(row["series_uid"]))
        return index.get(key, [])

    # --- Part 1/2: MULTIPLE_ACQUISITIONS re-analysis ---------------
    ma_rows = usability_df[usability_df["usability_category"] == "MULTIPLE_ACQUISITIONS"]
    print(f"\nAnalyzing {len(ma_rows)} MULTIPLE_ACQUISITIONS series...")
    ma_results = []
    for _, row in ma_rows.iterrows():
        files = get_files(row)
        if not files:
            continue
        geo = geometric_analysis(files)
        rec_class, evidence = classify_multiple_acquisition_series(geo)
        ma_results.append({
            "patient_id": row["patient_id"], "class": row.get("class"), "split": row.get("split"),
            "study_instance_uid": row["study_uid"], "series_instance_uid": row["series_uid"],
            "number_of_slices": geo["number_of_slices"],
            "unique_acquisition_numbers": geo["unique_acquisition_numbers"],
            "unique_acquisition_times": geo["unique_acquisition_times"],
            "unique_series_numbers": geo["unique_series_numbers"],
            "unique_orientations": geo["unique_orientations"],
            "unique_pixel_spacings": geo["unique_pixel_spacings"],
            "unique_dimensions": geo["unique_dimensions"],
            "duplicate_z_positions": geo["duplicate_z_positions"],
            "min_z": geo["min_z"], "max_z": geo["max_z"],
            "median_spacing": geo["median_spacing"],
            "detected_sweep_count": geo["detected_sweep_count"],
            "sweep_lengths": str(geo["sweep_lengths"]),
            "spatially_coherent": geo["spatially_coherent"],
            "recommended_classification": rec_class,
            "evidence_reason": evidence,
        })
    ma_recommendation_counts = Counter(r["recommended_classification"] for r in ma_results)
    print(f"  Recommended classifications: {dict(ma_recommendation_counts)}")

    # --- Part 3: USABLE control group -------------------------------
    usable_rows = usability_df[usability_df["usability_category"] == "USABLE"]
    print(f"\nAnalyzing {len(usable_rows)} USABLE series as control group...")
    usable_stats_list = []
    for _, row in usable_rows.iterrows():
        files = get_files(row)
        if not files:
            continue
        geo = geometric_analysis(files)
        usable_stats_list.append(geo)

    def agg_stats(values):
        arr = np.array([v for v in values if v is not None])
        if len(arr) == 0:
            return None
        return {"median": float(np.median(arr)), "mean": float(np.mean(arr)),
                "min": float(np.min(arr)), "max": float(np.max(arr))}

    usable_control = {
        "count_analyzed": len(usable_stats_list),
        "slice_count": agg_stats([g["number_of_slices"] for g in usable_stats_list]),
        "median_spacing": agg_stats([g["median_spacing"] for g in usable_stats_list]),
        "unique_acquisition_numbers": agg_stats([g["unique_acquisition_numbers"] for g in usable_stats_list]),
        "unique_pixel_spacings": agg_stats([g["unique_pixel_spacings"] for g in usable_stats_list]),
        "unique_orientations": agg_stats([g["unique_orientations"] for g in usable_stats_list]),
        "duplicate_z_positions_present_pct": (
            sum(1 for g in usable_stats_list if g["duplicate_z_positions"] > 0) /
            len(usable_stats_list) * 100 if usable_stats_list else None
        ),
        "multi_acquisition_number_pct": (
            sum(1 for g in usable_stats_list if g["unique_acquisition_numbers"] > 1) /
            len(usable_stats_list) * 100 if usable_stats_list else None
        ),
    }
    with open(os.path.join(args.output, "usable_control_statistics.json"), "w") as f:
        json.dump(usable_control, f, indent=2, default=str)

    # --- Part 4: OTHER investigation ---------------------------------
    other_rows = usability_df[usability_df["usability_category"] == "OTHER"]
    print(f"\nAnalyzing {len(other_rows)} OTHER series...")
    other_results = []
    for _, row in other_rows.iterrows():
        files = get_files(row)
        if not files:
            other_results.append({
                "patient_id": row["patient_id"], "study_instance_uid": row["study_uid"],
                "series_instance_uid": row["series_uid"],
                "actual_reason": "Series not found in working dataset re-index - "
                                  "possible grouping-key mismatch.",
            })
            continue
        non_ct = [f for f in files if f["modality"] != "CT"]
        missing_pid = [f for f in files if not f["has_patient_id"]]
        missing_study = [f for f in files if not f["has_study_uid"]]
        missing_series = [f for f in files if not f["has_series_uid"]]
        if non_ct:
            reason = f"non_ct: {len(non_ct)}/{len(files)} slice(s) with Modality={set(f['modality'] for f in non_ct)}"
        elif missing_pid or missing_study or missing_series:
            reason = (f"missing_required_metadata: missing_patient_id={len(missing_pid)}, "
                      f"missing_study_uid={len(missing_study)}, missing_series_uid={len(missing_series)}")
        else:
            reason = "unexpected_dicom_structure: did not match any known OTHER sub-reason - needs manual review"
        other_results.append({
            "patient_id": row["patient_id"], "study_instance_uid": row["study_uid"],
            "series_instance_uid": row["series_uid"], "number_of_files": len(files),
            "actual_reason": reason,
        })

    # --- Part 5: INCONSISTENT_PIXEL_SPACING investigation -----------
    ps_rows = usability_df[usability_df["usability_category"] == "INCONSISTENT_PIXEL_SPACING"]
    print(f"\nAnalyzing {len(ps_rows)} INCONSISTENT_PIXEL_SPACING series...")
    ps_results = []
    for _, row in ps_rows.iterrows():
        files = get_files(row)
        if not files:
            continue
        spacings = [tuple(f["pixel_spacing"]) for f in files if f["pixel_spacing"]]
        unique_spacings = sorted(set(spacings))
        dims = set((f["rows"], f["columns"]) for f in files)
        orientations = set(tuple(f["image_orientation"]) for f in files if f["image_orientation"])

        if len(unique_spacings) >= 2:
            vals = [s[0] for s in unique_spacings]
            rel_diff = (max(vals) - min(vals)) / min(vals) if min(vals) else float("inf")
            noise_only = rel_diff <= PIXEL_SPACING_NOISE_TOLERANCE
        else:
            rel_diff = 0.0
            noise_only = True

        also_dims_change = len(dims) > 1
        also_orientation_change = len(orientations) > 1

        if noise_only and not also_dims_change:
            category = "LIKELY_ASSESSMENT_FALSE_POSITIVE"
        elif also_dims_change or also_orientation_change:
            category = "CLEARLY_UNUSABLE"
        else:
            category = "POSSIBLY_USABLE_REQUIRES_REVIEW"

        ps_results.append({
            "patient_id": row["patient_id"], "study_instance_uid": row["study_uid"],
            "series_instance_uid": row["series_uid"],
            "unique_pixel_spacing_values": str(unique_spacings),
            "relative_difference_pct": rel_diff * 100,
            "dimensions_also_change": also_dims_change,
            "orientation_also_changes": also_orientation_change,
            "category": category,
        })

    # --- Part 6: INSUFFICIENT_SLICES investigation -------------------
    is_rows = usability_df[usability_df["usability_category"] == "INSUFFICIENT_SLICES"]
    print(f"\nAnalyzing {len(is_rows)} INSUFFICIENT_SLICES series...")
    patient_study_series_count = Counter(
        (str(r["patient_id"]), str(r["study_uid"])) for _, r in usability_df.iterrows()
    )
    is_results = []
    for _, row in is_rows.iterrows():
        files = get_files(row)
        related = patient_study_series_count.get((str(row["patient_id"]), str(row["study_uid"])), 0)
        is_results.append({
            "patient_id": row["patient_id"], "class": row.get("class"), "split": row.get("split"),
            "series_instance_uid": row["series_uid"],
            "number_of_slices": len(files),
            "related_series_for_same_patient_study": related - 1,  # excluding itself
        })

    # --- Part 7: comparison stats ------------------------------------
    ma_spacings = [r["median_spacing"] for r in ma_results if r["median_spacing"] is not None]
    us_spacings = [g["median_spacing"] for g in usable_stats_list if g["median_spacing"] is not None]
    ma_slice_counts = [r["number_of_slices"] for r in ma_results]
    us_slice_counts = [g["number_of_slices"] for g in usable_stats_list]

    comparison = {
        "usable_median_slice_count": float(np.median(us_slice_counts)) if us_slice_counts else None,
        "multiple_acquisition_median_slice_count": float(np.median(ma_slice_counts)) if ma_slice_counts else None,
        "usable_median_spacing": float(np.median(us_spacings)) if us_spacings else None,
        "multiple_acquisition_median_spacing": float(np.median(ma_spacings)) if ma_spacings else None,
        "usable_duplicate_z_rate_pct": usable_control["duplicate_z_positions_present_pct"],
        "multiple_acquisition_duplicate_z_rate_pct": (
            sum(1 for r in ma_results if r["duplicate_z_positions"] > 0) / len(ma_results) * 100
            if ma_results else None
        ),
        "usable_multi_acquisition_number_rate_pct": usable_control["multi_acquisition_number_pct"],
        "multiple_acquisition_multi_acquisition_number_rate_pct": (
            sum(1 for r in ma_results if r["unique_acquisition_numbers"] > 1) / len(ma_results) * 100
            if ma_results else None
        ),
    }

    # --- Save all outputs ---------------------------------------------
    summary = {
        "usable_series_analyzed": len(usable_stats_list),
        "multiple_acquisition_series_analyzed": len(ma_results),
        "pixel_spacing_series_analyzed": len(ps_results),
        "other_series_analyzed": len(other_results),
        "insufficient_slices_series_analyzed": len(is_results),
        "multiple_acquisition_recommended_classification_counts": dict(ma_recommendation_counts),
        "pixel_spacing_category_counts": dict(Counter(r["category"] for r in ps_results)),
        "comparison_usable_vs_multiple_acquisition": comparison,
    }
    with open(os.path.join(args.output, "usability_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    pd.DataFrame(ma_results).to_csv(os.path.join(args.output, "multiple_acquisition_validation.csv"), index=False)
    pd.DataFrame(ps_results).to_csv(os.path.join(args.output, "pixel_spacing_validation.csv"), index=False)
    pd.DataFrame(other_results).to_csv(os.path.join(args.output, "other_series_validation.csv"), index=False)
    pd.DataFrame(is_results).to_csv(os.path.join(args.output, "insufficient_slices_validation.csv"), index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))

    print(f"\nReports saved to: {args.output}")
    print("Working dataset was NOT modified - this script only read files.")
    print("Original assessment (assess_dicom_3d_usability.py and its output) "
          "was NOT changed - this is an independent, advisory validation only.")


if __name__ == "__main__":
    main()
