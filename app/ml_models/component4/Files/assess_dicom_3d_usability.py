"""
assess_dicom_3d_usability.py

STANDALONE, READ-ONLY series-level 3D usability assessment.

IMPORTANT, DELIBERATE DIFFERENCE FROM series.py/volume.py: this script
prefers ImagePositionPatient (true physical z-position) over
InstanceNumber when BOTH are available on every slice - the opposite
priority from series.py's _sort_slices() and volume.py's
determine_ordering(), which try InstanceNumber first. This is
intentional, per explicit instruction for this assessment: a series
should not be rejected just because InstanceNumber is out of physical
order if ImagePositionPatient reliably establishes the true spatial
sequence. This script does NOT modify, import, or depend on series.py
or volume.py in any way - it is fully independent, so this different
ordering strategy coexisting with the production code's own priority
is safe and does not create any inconsistency in the production path.

CRITICAL RULE, PART E: a physical gap between slices is NEVER, by
itself, a rejection reason. Gap/spacing statistics are recorded for
every USABLE series (median/min/max spacing) purely for reporting -
they never factor into the usable/unusable decision anywhere in this
script.

READ-ONLY. Uses stop_before_pixels=True throughout. Never calls
shutil.copy/move, os.rename, os.remove, Path.unlink, or any equivalent
write/delete operation against the working dataset. The only writes
this script performs are the three report files inside --output.

Does NOT organize files into usable/unusable folders - assessment
report only, per explicit instruction (that is a separate, later step).

Usage:
    python assess_dicom_3d_usability.py \
        --dataset-root "D:/DICOM/archive/dicom_3d_working_dataset" \
        --manifest "D:/DICOM/archive/dicom_3d_dataset/dicom_3d_manifest.csv" \
        --output "D:/DICOM/archive/dicom_3d_usability"
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

# --- Documented, reasoned thresholds ------------------------------------

# Two z-positions are treated as "duplicate" if they differ by less
# than this, in mm - accounts for real floating-point noise in
# ImagePositionPatient without treating genuinely close-but-different
# slices as duplicates.
DUPLICATE_POSITION_TOLERANCE_MM = 0.01

# Standard axial orientation cosines, same value/reasoning used
# throughout this project's other DICOM tooling.
STANDARD_AXIAL_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
AXIAL_TOLERANCE = 0.05

# Minimum length for a monotonic run to count as a real "sweep" rather
# than noise/a single misplaced slice - same reasoning as the prior
# multi-acquisition detection work in this project.
MIN_SWEEP_LENGTH = 3


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

def build_index(dataset_root: str) -> tuple[dict, int, int]:
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
            if not (patient_id and study_uid and series_uid):
                continue

            meta = {
                "path": fpath,
                "folder_class": folder,
                "modality": getattr(ds, "Modality", None),
                "instance_number": int(getattr(ds, "InstanceNumber")) if hasattr(ds, "InstanceNumber") else None,
                "image_position": [float(v) for v in ds.ImagePositionPatient] if hasattr(ds, "ImagePositionPatient") else None,
                "image_orientation": [float(v) for v in ds.ImageOrientationPatient] if hasattr(ds, "ImageOrientationPatient") else None,
                "pixel_spacing": [float(v) for v in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else None,
                "rows": int(ds.Rows) if hasattr(ds, "Rows") else None,
                "columns": int(ds.Columns) if hasattr(ds, "Columns") else None,
                "slice_thickness": float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else None,
                "spacing_between_slices": float(ds.SpacingBetweenSlices) if hasattr(ds, "SpacingBetweenSlices") else None,
                "acquisition_number": str(getattr(ds, "AcquisitionNumber", "")) or None,
                "acquisition_time": str(getattr(ds, "AcquisitionTime", "")) or None,
                "series_number": str(getattr(ds, "SeriesNumber", "")) or None,
            }
            key = (str(patient_id), str(study_uid), str(series_uid))
            index[key].append(meta)

    print(f"Indexed {total} files ({unreadable} unreadable). "
          f"{len(index)} distinct series found.")
    return dict(index), total, unreadable


# ---------------------------------------------------------------------
# Step 2: per-series assessment
# ---------------------------------------------------------------------

def assess_series(key: tuple, files: list[dict], manifest_lookup: dict) -> dict:
    patient_id, study_uid, series_uid = key
    n = len(files)

    result = {
        "patient_id": patient_id, "study_uid": study_uid, "series_uid": series_uid,
        "number_of_slices": n,
    }
    mf = manifest_lookup.get(key, {})
    result["split"] = mf.get("split")
    result["class"] = mf.get("class")

    first = files[0]
    result["rows"] = first.get("rows")
    result["columns"] = first.get("columns")
    result["pixel_spacing_x"] = first["pixel_spacing"][0] if first.get("pixel_spacing") else None
    result["pixel_spacing_y"] = first["pixel_spacing"][1] if first.get("pixel_spacing") else None
    result["slice_thickness"] = first.get("slice_thickness")
    result["spacing_between_slices"] = first.get("spacing_between_slices")
    result["acquisition_number"] = first.get("acquisition_number")
    result["acquisition_time"] = first.get("acquisition_time")
    result["series_number"] = first.get("series_number")

    def reject(category: str, reason: str) -> dict:
        result["usable_3d"] = False
        result["usability_reason"] = reason
        result["usability_category"] = category
        return result

    # --- Check 1: Modality ------------------------------------------
    non_ct = [f for f in files if f["modality"] != "CT"]
    if non_ct:
        return reject("OTHER", f"{len(non_ct)}/{n} slice(s) have Modality != 'CT'.")

    # --- Check 2: minimum slice count --------------------------------
    if n < 2:
        return reject("INSUFFICIENT_SLICES", f"Only {n} slice(s) - a volume requires at least 2.")

    # --- Check 3-4: consistent Rows/Columns --------------------------
    dims = set((f["rows"], f["columns"]) for f in files)
    if len(dims) > 1 or None in [d for pair in dims for d in pair]:
        return reject("INCONSISTENT_DIMENSIONS", f"Inconsistent or missing Rows/Columns: {dims}")

    # --- Check 5: consistent PixelSpacing ----------------------------
    if any(f["pixel_spacing"] is None for f in files):
        return reject("INCONSISTENT_PIXEL_SPACING", "PixelSpacing missing on at least one slice.")
    pixel_spacings = set(tuple(f["pixel_spacing"]) for f in files)
    if len(pixel_spacings) > 1:
        return reject("INCONSISTENT_PIXEL_SPACING", f"Inconsistent PixelSpacing across slices: {pixel_spacings}")

    # --- Check 6: orientation exists and is compatible ---------------
    if any(f["image_orientation"] is None for f in files):
        return reject("INCONSISTENT_ORIENTATION", "ImageOrientationPatient missing on at least one slice.")
    orientations_axial = [is_standard_axial(f["image_orientation"]) for f in files]
    if not all(orientations_axial):
        return reject("INCONSISTENT_ORIENTATION",
                       f"{sum(1 for o in orientations_axial if not o)}/{n} slice(s) have "
                       f"non-standard-axial orientation.")

    # --- Check 7-8: position + ordering. PREFERS ImagePositionPatient
    # over InstanceNumber, per Part F - a deliberate, documented
    # difference from series.py/volume.py's own priority. ------------
    has_all_position = all(f["image_position"] is not None for f in files)
    has_all_instance = all(f["instance_number"] is not None for f in files)

    if has_all_position:
        ordered = sorted(files, key=lambda f: f["image_position"][2])
        ordering_method = "ImagePositionPatient"
    elif has_all_instance:
        ordered = sorted(files, key=lambda f: f["instance_number"])
        ordering_method = "InstanceNumber"
        result["_no_position_warning"] = True
    else:
        return reject("UNRELIABLE_ORDERING",
                       "Neither ImagePositionPatient nor InstanceNumber is present on "
                       "every slice - no reliable ordering available.")

    result["ordering_method"] = ordering_method

    if not has_all_position:
        # Position-based checks below (duplicates, multi-acquisition,
        # gap stats) require position - can't be performed without it,
        # even though InstanceNumber gave us AN order.
        return reject("MISSING_SPATIAL_POSITION",
                       "ImagePositionPatient missing on at least one slice - ordering "
                       "fell back to InstanceNumber, but spatial validation "
                       "(duplicate positions, gap analysis) requires real positions.")

    z = np.array([f["image_position"][2] for f in ordered])
    result["min_z"] = float(z.min())
    result["max_z"] = float(z.max())

    # --- Check 10-11: multiple acquisitions - checked BEFORE duplicate
    # positions, since a systemic multi-sweep pattern is the more
    # specific/severe finding. IMPORTANT: the monotonic-run check must
    # use the ORIGINAL RECORDED sequence (InstanceNumber order), NOT
    # the position-sorted order used above - sorting by position
    # trivially collapses two overlapping sweeps into one non-decreasing
    # sequence with many exact ties, masking the real multi-sweep
    # signature. This was caught by testing (a synthetic 2-sweep series
    # was initially misclassified as DUPLICATE_SLICE_POSITIONS before
    # this fix) rather than assumed correct.
    acquisition_numbers = set(f["acquisition_number"] for f in files if f["acquisition_number"])
    acquisition_times = set(f["acquisition_time"] for f in files if f["acquisition_time"])

    multi_acq_evidence = []
    if has_all_instance:
        instance_ordered = sorted(files, key=lambda f: f["instance_number"])
        z_by_instance = [f["image_position"][2] for f in instance_ordered]
        run_lengths = count_monotonic_runs(z_by_instance)
        long_runs = [r for r in run_lengths if r >= MIN_SWEEP_LENGTH]
        if len(long_runs) >= 2:
            multi_acq_evidence.append(
                f"{len(long_runs)} separate long monotonic z-runs in recorded "
                f"InstanceNumber order (lengths {long_runs})"
            )
    if len(acquisition_numbers) > 1:
        multi_acq_evidence.append(f"{len(acquisition_numbers)} distinct AcquisitionNumber values")
    if len(acquisition_times) > 1:
        multi_acq_evidence.append(f"{len(acquisition_times)} distinct AcquisitionTime values")

    if multi_acq_evidence:
        return reject("MULTIPLE_ACQUISITIONS", "; ".join(multi_acq_evidence))

    # --- Check 9: duplicate physical slice positions ------------------
    diffs = np.abs(np.diff(z))
    duplicate_count = int((diffs < DUPLICATE_POSITION_TOLERANCE_MM).sum())
    if duplicate_count > 0:
        return reject("DUPLICATE_SLICE_POSITIONS",
                       f"{duplicate_count} pair(s) of slices share the same physical "
                       f"z-position (within {DUPLICATE_POSITION_TOLERANCE_MM}mm tolerance).")

    # --- Check 12: passed everything - USABLE. Gap size, per Part E,
    # is recorded for reporting only and NEVER causes rejection. ------
    if len(diffs) > 0:
        result["median_spacing"] = float(np.median(diffs))
        result["minimum_spacing"] = float(diffs.min())
        result["maximum_spacing"] = float(diffs.max())
    else:
        result["median_spacing"] = result["minimum_spacing"] = result["maximum_spacing"] = None

    result["usable_3d"] = True
    result["usability_reason"] = "All geometry checks passed."
    result["usability_category"] = "USABLE"
    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def load_manifest_lookup(manifest_path: str | None) -> dict:
    if not manifest_path or not os.path.isfile(manifest_path):
        print("No manifest provided or found - split/class will be blank in the report "
              "(the working copy is the source of truth for the assessment itself; "
              "the manifest is only used to enrich split/class labels where available).")
        return {}
    df = pd.read_csv(manifest_path)
    lookup = {}
    for _, row in df.iterrows():
        key = (str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))
        lookup[key] = {"split": row.get("split"), "class": row.get("class")}
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                         help="The WORKING COPY - source of truth for this assessment.")
    parser.add_argument("--manifest", default=None,
                         help="Optional, supporting only - not assumed complete.")
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_usability")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("DICOM 3D USABILITY ASSESSMENT - READ-ONLY, ASSESSMENT ONLY")
    print("=" * 70)

    manifest_lookup = load_manifest_lookup(args.manifest)
    index, total_files, unreadable = build_index(args.dataset_root)

    print(f"\nAssessing {len(index)} series...")
    results = []
    for i, (key, files) in enumerate(index.items()):
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(index)}")
        results.append(assess_series(key, files, manifest_lookup))

    usable = [r for r in results if r["usable_3d"]]
    unusable = [r for r in results if not r["usable_3d"]]

    category_counts = Counter(r["usability_category"] for r in results)
    class_counts = Counter(r.get("class") for r in results)
    split_counts = Counter(r.get("split") for r in results)
    total_patients = len(set(r["patient_id"] for r in results))
    total_studies = len(set((r["patient_id"], r["study_uid"]) for r in results))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total series: {len(results)}")
    print(f"Usable: {len(usable)} ({len(usable)/len(results)*100:.2f}%)")
    print(f"Unusable: {len(unusable)} ({len(unusable)/len(results)*100:.2f}%)")
    print(f"\nCategory breakdown:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")
    print(f"\nTotal DICOM files assessed: {total_files}")
    print(f"Total patients: {total_patients}")
    print(f"Total studies: {total_studies}")

    summary = {
        "total_series": len(results),
        "usable_series": len(usable),
        "unusable_series": len(unusable),
        "usable_pct": len(usable) / len(results) * 100 if results else None,
        "unusable_pct": len(unusable) / len(results) * 100 if results else None,
        "category_counts": dict(category_counts),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "total_files_assessed": total_files,
        "unreadable_files": unreadable,
        "total_patients": total_patients,
        "total_studies": total_studies,
    }
    with open(os.path.join(args.output, "3d_usability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    cols = ["split", "class", "patient_id", "study_uid", "series_uid",
            "number_of_slices", "usable_3d", "usability_reason", "usability_category",
            "ordering_method", "min_z", "max_z", "median_spacing", "minimum_spacing",
            "maximum_spacing", "rows", "columns", "pixel_spacing_x", "pixel_spacing_y",
            "slice_thickness", "spacing_between_slices", "acquisition_number",
            "acquisition_time", "series_number"]
    full_df = pd.DataFrame(results)
    for c in cols:
        if c not in full_df.columns:
            full_df[c] = None
    full_df[cols].to_csv(os.path.join(args.output, "3d_series_usability.csv"), index=False)

    unusable_df = pd.DataFrame(unusable)
    if len(unusable_df):
        for c in cols:
            if c not in unusable_df.columns:
                unusable_df[c] = None
        unusable_df[cols].to_csv(os.path.join(args.output, "3d_unusable_series.csv"), index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(os.path.join(args.output, "3d_unusable_series.csv"), index=False)

    print(f"\nReports saved to: {args.output}")
    print("Working dataset was NOT modified - this script only read files.")
    print("No usable/unusable folders were created - assessment report only, as instructed.")


if __name__ == "__main__":
    main()
