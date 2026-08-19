"""
validate_real_dicom_volumes.py

Standalone, READ-ONLY validation of the real Phase 1 build_volume()
implementation against every series in the organized V3 dataset. Does
NOT reimplement geometry validation, ordering, or HU conversion -
imports and calls the real project functions unchanged.

Reads dicom_3d_manifest.csv (already produced by the organizer script)
to determine series membership - does not re-scan or re-derive
grouping independently, since the manifest is the authoritative record
of what was actually organized.

For each series: reads the DICOM files listed under the manifest's
new_path column (inside dicom_3d_dataset - never touches the original
imbalanced_dataset), passes the RAW, UNSORTED list of pydicom.Dataset
objects to the real build_volume(), and records PASS/FAIL with full
detail. build_volume() does its own internal ordering and geometry
validation - this script never bypasses or duplicates that logic.

READ-ONLY: never writes, copies, moves, renames, or deletes anything
under dicom_3d_dataset except the one output report CSV, which is
written to the path you specify (default: alongside the manifest).

Usage:
    python validate_real_dicom_volumes.py --dataset-root "D:/DICOM/archive/dicom_3d_dataset"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

DEFAULT_PROJECT_ROOT = r"D:\R26-IT-065"

# Substrings matched against the REAL VolumeGeometryError messages
# raised by build_volume() (see volume.py) - used only to bucket
# failures for the summary printout below. The underlying PASS/FAIL
# decision itself always comes directly from build_volume() catching
# VolumeGeometryError - this categorization never influences that
# decision, it only labels it afterward for readability.
FAILURE_CATEGORIES = [
    ("missing SeriesInstanceUID", "missing SeriesInstanceUID"),
    ("different SeriesInstanceUID", "mixed series (multiple SeriesInstanceUID)"),
    ("No reliable slice ordering", "ordering failure"),
    ("missing ImagePositionPatient", "missing ImagePositionPatient"),
    ("not monotonic", "non-monotonic z-ordering"),
    ("inter-slice spacing is zero", "zero inter-slice spacing"),
    ("Inter-slice spacing is inconsistent", "inconsistent spacing"),
    ("missing PixelSpacing", "missing PixelSpacing"),
    ("PixelSpacing is not consistent", "inconsistent pixel spacing"),
    ("Inconsistent image dimensions", "inconsistent dimensions"),
    ("missing ImageOrientationPatient", "missing ImageOrientationPatient"),
    ("non-standard-axial orientation", "non-axial orientation"),
    ("Stacked volume shape", "internal shape mismatch"),
    ("at least 2 slices", "insufficient slices"),
    ("No slices provided", "no slices"),
]


def categorize_failure(message: str) -> str:
    for substring, label in FAILURE_CATEGORIES:
        if substring in message:
            return label
    return "other/uncategorized"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                         help="Path to dicom_3d_dataset (containing "
                              "dicom_3d_manifest.csv).")
    parser.add_argument("--manifest", default=None,
                         help="Defaults to <dataset-root>/dicom_3d_manifest.csv")
    parser.add_argument("--output", default=None,
                         help="Defaults to <dataset-root>/v3_volume_validation_report.csv")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT,
                         help=f"Path to the project root containing 'app/'. "
                              f"Defaults to '{DEFAULT_PROJECT_ROOT}'.")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.dataset_root, "dicom_3d_manifest.csv")
    output_path = args.output or os.path.join(args.dataset_root, "v3_volume_validation_report.csv")

    if not os.path.isdir(args.project_root) or not os.path.isdir(os.path.join(args.project_root, "app")):
        print(f"ERROR: --project-root '{args.project_root}' is invalid "
              f"(no 'app' folder found there).", file=sys.stderr)
        sys.exit(1)
    project_root_abs = os.path.abspath(args.project_root)
    if project_root_abs not in sys.path:
        sys.path.insert(0, project_root_abs)

    # Import the REAL project functions - not reimplemented anywhere below.
    try:
        import pydicom
        from app.ml_models.component4.dicom.volume import build_volume, VolumeGeometryError
    except ImportError as exc:
        print(f"ERROR: could not import the real build_volume(). "
              f"Confirm --project-root is correct.\nOriginal error: {exc}",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("REAL DICOM VOLUME VALIDATION - READ-ONLY")
    print("=" * 70)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Using real build_volume() from app.ml_models.component4.dicom.volume")

    print("\nReading manifest and grouping by series identity "
          "(patient_id + study_instance_uid + series_instance_uid)...")
    series_groups = defaultdict(list)
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["patient_id"], row["study_instance_uid"], row["series_instance_uid"])
            series_groups[key].append(row)

    print(f"Total series found in manifest: {len(series_groups)}")

    results = []
    failure_categories = Counter()

    for i, (key, rows) in enumerate(series_groups.items()):
        patient_id, study_uid, series_uid = key
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(series_groups)} series processed")

        split = rows[0]["split"]
        cls = rows[0]["class"]
        n_slices_expected = len(rows)

        # Read the RAW datasets - unsorted, exactly as build_volume() expects.
        # Reads only from new_path (inside dicom_3d_dataset) - never touches
        # original_path / the source imbalanced_dataset.
        datasets = []
        read_errors = []
        for row in rows:
            path = row["new_path"]
            try:
                ds = pydicom.dcmread(path)  # full read - pixel data needed by build_volume()
                datasets.append(ds)
            except Exception as exc:
                read_errors.append((path, str(exc)))

        if read_errors:
            results.append({
                "split": split, "class": cls, "patient_id": patient_id,
                "study_instance_uid": study_uid, "series_instance_uid": series_uid,
                "num_slices": n_slices_expected, "result": "FAIL",
                "failure_reason": f"{len(read_errors)} file(s) unreadable: {read_errors[0][1][:100]}",
                "failure_category": "file read error",
                "volume_shape": "", "pixel_spacing": "", "inter_slice_spacing": "",
                "hu_min": "", "hu_max": "", "ordering_method": "",
            })
            failure_categories["file read error"] += 1
            continue

        try:
            volume = build_volume(datasets)  # REAL function, unmodified, called as-is
            results.append({
                "split": split, "class": cls, "patient_id": patient_id,
                "study_instance_uid": study_uid, "series_instance_uid": series_uid,
                "num_slices": n_slices_expected, "result": "PASS",
                "failure_reason": "", "failure_category": "",
                "volume_shape": str(volume.shape),
                "pixel_spacing": str(volume.pixel_spacing),
                "inter_slice_spacing": f"{volume.inter_slice_spacing:.6f}",
                "hu_min": f"{volume.volume.min():.2f}",
                "hu_max": f"{volume.volume.max():.2f}",
                "ordering_method": volume.ordering_method,
            })
        except VolumeGeometryError as exc:
            msg = str(exc)
            category = categorize_failure(msg)
            failure_categories[category] += 1
            results.append({
                "split": split, "class": cls, "patient_id": patient_id,
                "study_instance_uid": study_uid, "series_instance_uid": series_uid,
                "num_slices": n_slices_expected, "result": "FAIL",
                "failure_reason": msg, "failure_category": category,
                "volume_shape": "", "pixel_spacing": "", "inter_slice_spacing": "",
                "hu_min": "", "hu_max": "", "ordering_method": "",
            })
        except Exception as exc:
            # Genuinely unexpected error (not a VolumeGeometryError) -
            # reported distinctly, never silently conflated with an
            # expected geometry-validation rejection.
            failure_categories["UNEXPECTED ERROR (not VolumeGeometryError)"] += 1
            results.append({
                "split": split, "class": cls, "patient_id": patient_id,
                "study_instance_uid": study_uid, "series_instance_uid": series_uid,
                "num_slices": n_slices_expected, "result": "FAIL",
                "failure_reason": f"UNEXPECTED: {type(exc).__name__}: {exc}",
                "failure_category": "UNEXPECTED ERROR (not VolumeGeometryError)",
                "volume_shape": "", "pixel_spacing": "", "inter_slice_spacing": "",
                "hu_min": "", "hu_max": "", "ordering_method": "",
            })

    total = len(results)
    passed = sum(1 for r in results if r["result"] == "PASS")
    failed = total - passed

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total series: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass percentage: {passed/total*100:.2f}%" if total else "N/A")

    print("\nFailure reason summary:")
    if failure_categories:
        for label, count in failure_categories.most_common():
            print(f"  {label}: {count}")
    else:
        print("  (no failures)")

    with open(output_path, "w", newline="") as f:
        fieldnames = ["split", "class", "patient_id", "study_instance_uid",
                      "series_instance_uid", "num_slices", "result",
                      "failure_reason", "failure_category", "volume_shape",
                      "pixel_spacing", "inter_slice_spacing", "hu_min", "hu_max",
                      "ordering_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nReport saved to: {output_path}")
    print("Dataset was NOT modified - this script only read files.")


if __name__ == "__main__":
    main()
