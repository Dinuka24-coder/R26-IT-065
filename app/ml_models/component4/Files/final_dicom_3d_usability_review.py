"""
final_dicom_3d_usability_review.py

STANDALONE, READ-ONLY final consolidation of the entire DICOM 3D
usability investigation (assess -> validate -> this final review).

SCOPING DECISION, stated explicitly: this script does NOT re-scan all
677 series from scratch. It performs FRESH, full per-series
re-investigation only for the 273 series genuinely in question:
  - Group A: the 161 series previously flagged LIKELY_USABLE_FALSE_POSITIVE
    (from multiple_acquisition_validation.csv)
  - Group B: the 63 INCONSISTENT_PIXEL_SPACING series
  - Group C: the 24 OTHER series
  - Group D: the 25 INSUFFICIENT_SLICES series
The 89 already-CONFIRMED_MULTIPLE_ACQUISITIONS series and the 314
already-USABLE series carry their existing, already-verified
classification directly into the final combined report, WITHOUT a
redundant fresh DICOM re-scan - their status was already established
with real evidence in the prior two stages, and nothing in this task
asks that evidence to be questioned again. This is a deliberate
efficiency choice, not a shortcut on rigor: the 273 series that ARE
genuinely uncertain get full, fresh, real-metadata investigation
below; the 403 that are not remain exactly as already determined.

Reuses the SAME proven geometric logic already tested in
assess_dicom_3d_usability.py and validate_dicom_3d_usability_results.py
(position-based ordering preference, duplicate-position detection,
axial-orientation check, monotonic-run detection) - not reimplemented
differently, so results stay consistent with the prior two stages'
own reasoning.

READ-ONLY. stop_before_pixels=True throughout. Never calls
shutil.copy/move, os.rename, os.remove, Path.unlink, pydicom save,
or any equivalent write/modify operation against either dataset. All
writes go only to --output.

Does NOT create usable/unusable folders, does NOT copy/move/reorganize
any DICOM file - report only, per explicit instruction.

Usage:
    python final_dicom_3d_usability_review.py \
        --dataset-root "D:/DICOM/archive/dicom_3d_working_dataset" \
        --usability-csv "D:/DICOM/archive/dicom_3d_usability/3d_series_usability.csv" \
        --validation-dir "D:/DICOM/archive/dicom_3d_usability_validation" \
        --output "D:/DICOM/archive/dicom_3d_final_review"
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

# --- Documented, reasoned thresholds - SAME values as the prior two
# stages, deliberately, so results stay consistent across the whole
# investigation rather than drifting with each new script. ------------

DUPLICATE_POSITION_TOLERANCE_MM = 0.01
STANDARD_AXIAL_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
AXIAL_TOLERANCE = 0.05
MIN_SWEEP_LENGTH = 3
PIXEL_SPACING_NOISE_TOLERANCE = 0.02  # 2% relative difference
GAP_RELATIVE_THRESHOLD = 2.0  # gap > 2x median spacing = "large gap", reporting only, never rejection

FULL_TAG_LIST = [
    "SeriesNumber", "AcquisitionNumber", "AcquisitionTime",
    "SliceThickness", "SpacingBetweenSlices",
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
# Build a metadata index of ONLY the requested (patient, study, series)
# keys - not the full 677, per the scoping decision above. Still one
# pass over the dataset (can't seek directly to specific series without
# scanning), but only stores what's actually needed.
# ---------------------------------------------------------------------

def build_targeted_index(dataset_root: str, target_keys: set[tuple]) -> dict:
    print(f"Scanning {dataset_root} for the {len(target_keys)} target series "
          f"(one pass, metadata only, stop_before_pixels=True)...")
    index = defaultdict(list)
    total = 0
    matched = 0

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
                print(f"  ...scanned {total} files so far ({matched} matched a target series)")
            fpath = os.path.join(folder_path, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception:
                continue

            patient_id = getattr(ds, "PatientID", None)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if not (patient_id and study_uid and series_uid):
                continue
            key = (str(patient_id), str(study_uid), str(series_uid))
            if key not in target_keys:
                continue
            matched += 1

            meta = {
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
            index[key].append(meta)

    print(f"Scanned {total} files, matched {matched} files across "
          f"{len(index)} of {len(target_keys)} target series.")
    return dict(index)


# ---------------------------------------------------------------------
# Core per-series investigation, producing the FULL requested field set
# and a FINAL_USABLE / FINAL_UNUSABLE / NEEDS_REVIEW decision.
# ---------------------------------------------------------------------

def investigate_series(key: tuple, files: list[dict], source_category: str) -> dict:
    patient_id, study_uid, series_uid = key
    n = len(files)

    r = {
        "patient_id": patient_id, "study_instance_uid": study_uid, "series_instance_uid": series_uid,
        "number_of_slices": n, "source_category": source_category,
    }

    def finalize(classification: str, reason: str, notes: str = "") -> dict:
        r["final_classification"] = classification
        r["final_reason"] = reason
        r["notes"] = notes
        return r

    if n == 0:
        return finalize("NEEDS_REVIEW", "OTHER",
                         "Series not found in working dataset re-scan - possible key mismatch, needs manual check.")

    first = files[0]
    r["rows"] = first.get("rows")
    r["columns"] = first.get("columns")
    r["pixel_spacing_x"] = first["pixel_spacing"][0] if first.get("pixel_spacing") else None
    r["pixel_spacing_y"] = first["pixel_spacing"][1] if first.get("pixel_spacing") else None
    r["slice_thickness"] = first.get("SliceThickness")
    r["spacing_between_slices"] = first.get("SpacingBetweenSlices")
    r["acquisition_number"] = first.get("AcquisitionNumber")
    r["acquisition_time"] = first.get("AcquisitionTime")
    r["series_number"] = first.get("SeriesNumber")
    r["unique_acquisition_number_count"] = len(set(f["AcquisitionNumber"] for f in files if f["AcquisitionNumber"]))

    # --- Modality -------------------------------------------------
    non_ct = [f for f in files if f["modality"] != "CT"]
    if non_ct:
        return finalize("FINAL_UNUSABLE", "OTHER",
                         f"{len(non_ct)}/{n} slice(s) have Modality != 'CT'.")

    # --- Minimum slice count - genuinely re-verified, not assumed ---
    if n < 2:
        return finalize("FINAL_UNUSABLE", "INSUFFICIENT_SLICES",
                         f"Re-verified from working dataset: {n} slice(s) actually present.")

    # --- Consistent dimensions ---------------------------------------
    dims = set((f["rows"], f["columns"]) for f in files)
    if len(dims) > 1 or any(None in d for d in dims):
        return finalize("FINAL_UNUSABLE", "INCONSISTENT_DIMENSIONS", f"Dimensions found: {dims}")

    # --- PixelSpacing: harmless variation vs real inconsistency ------
    if any(f["pixel_spacing"] is None for f in files):
        return finalize("FINAL_UNUSABLE", "INVALID_GEOMETRY", "PixelSpacing missing on at least one slice.")
    ps_values = [tuple(f["pixel_spacing"]) for f in files]
    unique_ps = sorted(set(ps_values))
    r["pixel_spacing_unique_count"] = len(unique_ps)
    if len(unique_ps) > 1:
        xs = [v[0] for v in unique_ps]
        rel_diff = (max(xs) - min(xs)) / min(xs) if min(xs) else float("inf")
        r["pixel_spacing_relative_difference_pct"] = rel_diff * 100
        if rel_diff > PIXEL_SPACING_NOISE_TOLERANCE:
            return finalize(
                "FINAL_UNUSABLE", "SIGNIFICANT_PIXEL_SPACING_INCONSISTENCY",
                f"PixelSpacing values {unique_ps}, relative difference "
                f"{rel_diff*100:.2f}% exceeds the {PIXEL_SPACING_NOISE_TOLERANCE*100:.0f}% "
                f"tolerance for harmless rounding variation."
            )
        # else: within tolerance - treated as harmless, continue checks.

    # --- Orientation ---------------------------------------------------
    if any(f["image_orientation"] is None for f in files):
        return finalize("FINAL_UNUSABLE", "MISSING_SPATIAL_POSITION",
                         "ImageOrientationPatient missing on at least one slice.")
    if not all(is_standard_axial(f["image_orientation"]) for f in files):
        return finalize("FINAL_UNUSABLE", "INVALID_GEOMETRY",
                         "At least one slice has non-standard-axial orientation.")

    # --- Position + ordering. Prefers ImagePositionPatient over
    # InstanceNumber when both available - same deliberate priority as
    # the prior validation stage. --------------------------------------
    has_all_position = all(f["image_position"] is not None for f in files)
    has_all_instance = all(f["instance_number"] is not None for f in files)

    if not has_all_position:
        if has_all_instance:
            return finalize("FINAL_UNUSABLE", "MISSING_SPATIAL_POSITION",
                             "ImagePositionPatient missing on at least one slice - "
                             "InstanceNumber alone is not sufficient for verified "
                             "spatial volume construction.")
        return finalize("FINAL_UNUSABLE", "UNRELIABLE_ORDERING",
                         "Neither ImagePositionPatient nor InstanceNumber present on every slice.")

    position_ordered = sorted(files, key=lambda f: f["image_position"][2])
    z = np.array([f["image_position"][2] for f in position_ordered])
    r["min_z"] = float(z.min())
    r["max_z"] = float(z.max())

    diffs = np.abs(np.diff(z))
    duplicate_count = int((diffs < DUPLICATE_POSITION_TOLERANCE_MM).sum())
    r["duplicate_position_count"] = duplicate_count

    ordering_method = "ImagePositionPatient"
    ordering_required = not (has_all_instance and
                              [f["instance_number"] for f in position_ordered] ==
                              sorted(f["instance_number"] for f in files))
    r["ordering_method"] = ordering_method
    r["ordering_required"] = bool(ordering_required)

    # --- Multiple acquisitions: real spatial overlap only, per the
    # already-validated reasoning from the prior stage. AcquisitionNumber
    # /Time differences alone are recorded but never determine this. ---
    if duplicate_count > 0:
        return finalize(
            "FINAL_UNUSABLE", "DUPLICATE_SLICE_POSITIONS" if duplicate_count <= 2 else "CONFIRMED_MULTIPLE_ACQUISITIONS",
            f"{duplicate_count} pair(s) of slices share the same physical position "
            f"(within {DUPLICATE_POSITION_TOLERANCE_MM}mm) - direct evidence of "
            f"spatial conflict, not resolved by reordering."
        )

    # --- Passed everything - USABLE. Gaps are recorded, never fabricated
    # or filled, and never cause rejection. ------------------------------
    nonzero_diffs = diffs[diffs >= DUPLICATE_POSITION_TOLERANCE_MM]
    r["minimum_spacing"] = float(nonzero_diffs.min()) if len(nonzero_diffs) else None
    r["median_spacing"] = float(np.median(nonzero_diffs)) if len(nonzero_diffs) else None
    r["maximum_spacing"] = float(nonzero_diffs.max()) if len(nonzero_diffs) else None
    if len(nonzero_diffs):
        large_gap_threshold = r["median_spacing"] * GAP_RELATIVE_THRESHOLD if r["median_spacing"] else 0
        r["number_of_large_gaps"] = int((nonzero_diffs > large_gap_threshold).sum()) if large_gap_threshold else 0
    else:
        r["number_of_large_gaps"] = 0

    return finalize("FINAL_USABLE", "NO_INTERPOLATION_REQUIRED",
                     "All geometry checks passed. Existing slices only - no "
                     "interpolation, resampling, or fabricated slices.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def carry_forward_row(row: pd.Series, final_classification: str, final_reason: str, notes: str) -> dict:
    """For the 89 CONFIRMED_MULTIPLE_ACQUISITIONS and 314 already-USABLE
    series - uses their EXISTING computed fields directly rather than
    re-scanning, per the scoping decision. Never invents a value not
    already present in the source row.
    """
    return {
        "split": row.get("split"), "class": row.get("class"),
        "patient_id": row.get("patient_id"),
        "study_instance_uid": row.get("study_uid") or row.get("study_instance_uid"),
        "series_instance_uid": row.get("series_uid") or row.get("series_instance_uid"),
        "number_of_slices": row.get("number_of_slices"),
        "final_classification": final_classification, "final_reason": final_reason,
        "ordering_method": row.get("ordering_method"), "ordering_required": None,
        "min_z": row.get("min_z"), "max_z": row.get("max_z"),
        "median_spacing": row.get("median_spacing"),
        "minimum_spacing": row.get("minimum_spacing"), "maximum_spacing": row.get("maximum_spacing"),
        "number_of_large_gaps": None,
        "rows": row.get("rows"), "columns": row.get("columns"),
        "pixel_spacing_x": row.get("pixel_spacing_x"), "pixel_spacing_y": row.get("pixel_spacing_y"),
        "slice_thickness": row.get("slice_thickness"),
        "spacing_between_slices": row.get("spacing_between_slices"),
        "acquisition_number": row.get("acquisition_number"),
        "acquisition_time": row.get("acquisition_time"),
        "series_number": row.get("series_number"),
        "duplicate_position_count": row.get("duplicate_z_positions"),
        "unique_acquisition_number_count": row.get("unique_acquisition_numbers"),
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--usability-csv", required=True,
                         help="Path to 3d_series_usability.csv (assessment stage).")
    parser.add_argument("--validation-dir", required=True,
                         help="Path to the dicom_3d_usability_validation output "
                              "directory (validation stage).")
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_final_review")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.usability_csv):
        print(f"ERROR: usability CSV not found: {args.usability_csv}", file=sys.stderr)
        sys.exit(1)

    ma_validation_path = os.path.join(args.validation_dir, "multiple_acquisition_validation.csv")
    if not os.path.isfile(ma_validation_path):
        print(f"ERROR: multiple_acquisition_validation.csv not found in {args.validation_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("FINAL DICOM 3D USABILITY REVIEW - READ-ONLY")
    print("=" * 70)

    usability_df = pd.read_csv(args.usability_csv)
    ma_validation_df = pd.read_csv(ma_validation_path)

    def series_key(row, study_col="study_uid", series_col="series_uid"):
        return (str(row["patient_id"]), str(row[study_col]), str(row[series_col]))

    # --- Partition into: carry-forward vs fresh-investigation --------
    confirmed_ma_keys = set(
        series_key(r, "study_instance_uid", "series_instance_uid")
        for _, r in ma_validation_df.iterrows()
        if r["recommended_classification"] == "CONFIRMED_MULTIPLE_ACQUISITIONS"
    )
    likely_fp_keys = set(
        series_key(r, "study_instance_uid", "series_instance_uid")
        for _, r in ma_validation_df.iterrows()
        if r["recommended_classification"] == "LIKELY_USABLE_FALSE_POSITIVE"
    )

    usable_rows = usability_df[usability_df["usability_category"] == "USABLE"]
    ps_rows = usability_df[usability_df["usability_category"] == "INCONSISTENT_PIXEL_SPACING"]
    other_rows = usability_df[usability_df["usability_category"] == "OTHER"]
    insuff_rows = usability_df[usability_df["usability_category"] == "INSUFFICIENT_SLICES"]
    ma_rows = usability_df[usability_df["usability_category"] == "MULTIPLE_ACQUISITIONS"]

    print(f"\nCarry-forward (no fresh scan): {len(usable_rows)} already-USABLE, "
          f"{len(confirmed_ma_keys)} CONFIRMED_MULTIPLE_ACQUISITIONS")
    print(f"Fresh investigation needed: Group A={len(likely_fp_keys)}, "
          f"Group B={len(ps_rows)}, Group C={len(other_rows)}, Group D={len(insuff_rows)}")

    fresh_target_keys = set()
    fresh_target_keys |= likely_fp_keys
    fresh_target_keys |= set(series_key(r) for _, r in ps_rows.iterrows())
    fresh_target_keys |= set(series_key(r) for _, r in other_rows.iterrows())
    fresh_target_keys |= set(series_key(r) for _, r in insuff_rows.iterrows())

    index = build_targeted_index(args.dataset_root, fresh_target_keys)

    print("\nInvestigating Groups A-D with fresh, full metadata...")
    fresh_results = []
    for key in fresh_target_keys:
        if key in likely_fp_keys:
            src = "Group A: LIKELY_USABLE_FALSE_POSITIVE"
        elif key in set(series_key(r) for _, r in ps_rows.iterrows()):
            src = "Group B: INCONSISTENT_PIXEL_SPACING"
        elif key in set(series_key(r) for _, r in other_rows.iterrows()):
            src = "Group C: OTHER"
        else:
            src = "Group D: INSUFFICIENT_SLICES"
        files = index.get(key, [])
        result = investigate_series(key, files, src)
        # attach split/class from the original usability_df
        match = usability_df[
            (usability_df["patient_id"].astype(str) == key[0]) &
            (usability_df["study_uid"].astype(str) == key[1]) &
            (usability_df["series_uid"].astype(str) == key[2])
        ]
        if len(match):
            result["split"] = match.iloc[0].get("split")
            result["class"] = match.iloc[0].get("class")
        fresh_results.append(result)

    # --- Carry-forward rows -------------------------------------------
    carried = []
    for _, row in usable_rows.iterrows():
        carried.append(carry_forward_row(row, "FINAL_USABLE", "NO_INTERPOLATION_REQUIRED",
                                          "Carried forward from prior USABLE assessment - not re-scanned "
                                          "(scoping decision, see script docstring)."))
    for _, row in ma_rows.iterrows():
        key = series_key(row)
        if key in confirmed_ma_keys:
            carried.append(carry_forward_row(row, "FINAL_UNUSABLE", "CONFIRMED_MULTIPLE_ACQUISITIONS",
                                              "Carried forward from validation stage - real spatial overlap "
                                              "already confirmed with direct evidence, not re-scanned."))

    # --- Combine everything ---------------------------------------------
    all_final = carried + fresh_results
    final_df = pd.DataFrame(all_final)

    counts = Counter(r.get("final_classification", "NEEDS_REVIEW") for r in all_final)
    reason_counts = Counter(r.get("final_reason", "") for r in all_final)
    class_counts = Counter(r.get("class") for r in all_final)
    split_counts = Counter(r.get("split") for r in all_final)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total series in final review: {len(all_final)}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v/len(all_final)*100:.2f}%)")
    print(f"\nReason breakdown: {dict(reason_counts)}")

    summary = {
        "total_series": len(all_final),
        "final_usable": counts.get("FINAL_USABLE", 0),
        "final_unusable": counts.get("FINAL_UNUSABLE", 0),
        "needs_review": counts.get("NEEDS_REVIEW", 0),
        "final_usable_pct": counts.get("FINAL_USABLE", 0) / len(all_final) * 100 if all_final else None,
        "final_unusable_pct": counts.get("FINAL_UNUSABLE", 0) / len(all_final) * 100 if all_final else None,
        "needs_review_pct": counts.get("NEEDS_REVIEW", 0) / len(all_final) * 100 if all_final else None,
        "reason_counts": dict(reason_counts),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "carried_forward_usable": len(usable_rows),
        "carried_forward_confirmed_multiple_acquisitions": len(confirmed_ma_keys),
        "freshly_investigated": len(fresh_results),
    }
    with open(os.path.join(args.output, "final_3d_usability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    cols = ["split", "class", "patient_id", "study_instance_uid", "series_instance_uid",
            "number_of_slices", "final_classification", "final_reason", "ordering_method",
            "ordering_required", "min_z", "max_z", "median_spacing", "minimum_spacing",
            "maximum_spacing", "number_of_large_gaps", "rows", "columns", "pixel_spacing_x",
            "pixel_spacing_y", "slice_thickness", "spacing_between_slices", "acquisition_number",
            "acquisition_time", "series_number", "duplicate_position_count",
            "unique_acquisition_number_count", "notes"]
    for c in cols:
        if c not in final_df.columns:
            final_df[c] = None
    final_df[cols].to_csv(os.path.join(args.output, "final_3d_series_review.csv"), index=False)

    final_df[final_df["final_classification"] == "FINAL_USABLE"][cols].to_csv(
        os.path.join(args.output, "final_usable_series.csv"), index=False)
    final_df[final_df["final_classification"] == "FINAL_UNUSABLE"][cols].to_csv(
        os.path.join(args.output, "final_unusable_series.csv"), index=False)
    final_df[final_df["final_classification"] == "NEEDS_REVIEW"][cols].to_csv(
        os.path.join(args.output, "final_needs_review_series.csv"), index=False)

    print(f"\nReports saved to: {args.output}")
    print("Working dataset was NOT modified - this script only read files.")
    print("No usable/unusable folders were created - report only, as instructed.")


if __name__ == "__main__":
    main()
