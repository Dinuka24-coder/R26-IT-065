"""
validate_build_volume_on_final_dataset.py

STANDALONE, READ-ONLY validation of the REAL build_volume() against the
REAL organized dicom_3d_final dataset (category/class/Patient_<id>/
Series_<n>/*.dcm structure). Different from the earlier
validate_real_dicom_volumes.py, which targeted a different, earlier
manifest/path layout - this version reads dicom_3d_final_manifest.csv
directly and groups by (patient_id, study_instance_uid,
series_instance_uid), matching the real organized structure exactly.

Imports and calls the REAL build_volume() unmodified - does not
reimplement or bypass any geometry validation.

Explicitly does NOT promote needs_review series to usable, regardless
of what build_volume() reports for them - reports accept/reject only.

READ-ONLY with respect to the organized dataset and the working
dataset. Never writes, copies, moves, renames, or deletes any DICOM
file. Only writes its own CSV/console report to --output.

Usage:
    python validate_build_volume_on_final_dataset.py \
        --final-dataset "D:/DICOM/archive/dicom_3d_final" \
        --plan-manifest "D:/DICOM/archive/dicom_3d_final/dicom_3d_final_manifest.csv" \
        --project-root "D:/R26-IT-065" \
        --output "D:/DICOM/archive/dicom_3d_volume_validation" \
        --sample-per-class 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pydicom

DEFAULT_PROJECT_ROOT = r"D:/R26-IT-065"

NAMED_CASES = {"Lung_Dx-A0216", "Lung_Dx-B0025", "Lung_Dx-A0198"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dataset", required=True)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_volume_validation")
    parser.add_argument("--sample-per-class", type=int, default=5,
                         help="Representative usable series to test per class, "
                              "in addition to ALL needs_review series and the "
                              "named cases, which are always fully tested.")
    args = parser.parse_args()

    if not os.path.isdir(args.project_root) or not os.path.isdir(os.path.join(args.project_root, "app")):
        print(f"ERROR: --project-root '{args.project_root}' invalid.", file=sys.stderr)
        sys.exit(1)
    project_root_abs = os.path.abspath(args.project_root)
    if project_root_abs not in sys.path:
        sys.path.insert(0, project_root_abs)

    try:
        from app.ml_models.component4.dicom.volume import build_volume, VolumeGeometryError
    except ImportError as exc:
        print(f"ERROR: could not import the real build_volume(): {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("build_volume() VALIDATION AGAINST THE REAL dicom_3d_final DATASET")
    print("=" * 70)

    plan_df = pd.read_csv(args.plan_manifest)
    print(f"Loaded plan manifest: {len(plan_df)} file rows")

    def skey(row):
        return (str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))

    series_groups = defaultdict(list)
    for _, row in plan_df.iterrows():
        series_groups[skey(row)].append(row)

    # --- Build the test selection: named cases (all their series) +
    # ALL needs_review + representative sample per class from usable ----
    to_test = {}  # key -> (category, class, patient_id)
    for key, rows in series_groups.items():
        first = rows[0]
        patient_id, study_uid, series_uid = key
        category = first["category"]
        cls = first["class"]
        if patient_id in NAMED_CASES:
            to_test[key] = (category, cls, patient_id, "named_case")
        elif category == "needs_review":
            to_test[key] = (category, cls, patient_id, "needs_review_full")

    by_class_usable = defaultdict(list)
    for key, rows in series_groups.items():
        if rows[0]["category"] == "usable":
            by_class_usable[rows[0]["class"]].append(key)
    for cls, keys in by_class_usable.items():
        for key in sorted(keys)[:args.sample_per_class]:
            if key not in to_test:
                to_test[key] = ("usable", cls, key[0], "usable_sample")

    print(f"\nSelected for testing: {len(to_test)} series")
    print(f"  Named cases: {sum(1 for v in to_test.values() if v[3]=='named_case')}")
    print(f"  ALL needs_review: {sum(1 for v in to_test.values() if v[3]=='needs_review_full')}")
    print(f"  Usable samples: {sum(1 for v in to_test.values() if v[3]=='usable_sample')}")

    results = []
    for key, (category, cls, patient_id, reason) in to_test.items():
        rows = series_groups[key]
        rows_sorted = sorted(rows, key=lambda r: int(r["slice_index"]))
        datasets = []
        read_errors = []
        for row in rows_sorted:
            try:
                # destination_path in the manifest is RELATIVE to
                # --final-dataset (e.g. "usable\small.cell.carcinoma\...")
                # and was written with Windows backslash separators - must
                # be explicitly resolved against the dataset root before
                # reading. Normalizing to forward slashes first makes this
                # unambiguous and correct via pathlib.PurePosixPath parsing
                # regardless of the separator style stored in the manifest,
                # rather than depending on OS-specific backslash handling.
                normalized = row["destination_path"].replace("\\", "/")
                full_path = Path(args.final_dataset) / normalized
                ds = pydicom.dcmread(str(full_path))
                datasets.append(ds)
            except Exception as exc:
                read_errors.append(str(exc))

        result = {
            "patient_id": patient_id, "class": cls, "category": category,
            "selection_reason": reason,
            "study_instance_uid": key[1], "series_instance_uid": key[2],
            "num_files_in_plan": len(rows), "num_files_read": len(datasets),
        }

        if read_errors:
            result["build_volume_result"] = "READ_ERROR"
            result["detail"] = f"{len(read_errors)} file(s) unreadable: {read_errors[0][:100]}"
            results.append(result)
            continue

        try:
            vol = build_volume(datasets)  # REAL function, unmodified
            result["build_volume_result"] = "ACCEPTED"
            result["volume_shape"] = str(vol.shape)
            result["pixel_spacing"] = str(vol.pixel_spacing)
            result["inter_slice_spacing"] = f"{vol.inter_slice_spacing:.4f}"
            result["hu_min"] = f"{vol.volume.min():.2f}"
            result["hu_max"] = f"{vol.volume.max():.2f}"
            result["ordering_method"] = vol.ordering_method
            result["detail"] = ""
        except VolumeGeometryError as exc:
            result["build_volume_result"] = "REJECTED"
            result["detail"] = str(exc)
        except Exception as exc:
            result["build_volume_result"] = "UNEXPECTED_ERROR"
            result["detail"] = f"{type(exc).__name__}: {exc}"

        results.append(result)

        marker = ""
        if patient_id in NAMED_CASES:
            marker = "  <-- NAMED CASE"
        elif category == "needs_review":
            marker = "  <-- needs_review (NOT promoted regardless of result)"
        print(f"  {patient_id} ({cls}, {category}): {result['build_volume_result']}{marker}")

    # --- Summary ------------------------------------------------------
    accepted = sum(1 for r in results if r["build_volume_result"] == "ACCEPTED")
    rejected = sum(1 for r in results if r["build_volume_result"] == "REJECTED")
    errors = sum(1 for r in results if r["build_volume_result"] in ("READ_ERROR", "UNEXPECTED_ERROR"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tested: {len(results)}")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Errors: {errors}")

    nr_results = [r for r in results if r["category"] == "needs_review"]
    print(f"\nneeds_review series tested: {len(nr_results)} "
          f"(accepted={sum(1 for r in nr_results if r['build_volume_result']=='ACCEPTED')}, "
          f"rejected={sum(1 for r in nr_results if r['build_volume_result']=='REJECTED')})")
    print("NONE of these are promoted to usable regardless of result - report only.")

    print("\nNamed case results:")
    for r in results:
        if r["patient_id"] in NAMED_CASES:
            print(f"  {r['patient_id']} / series {r['series_instance_uid'][-12:]}: "
                  f"{r['build_volume_result']} - {r['detail'][:80]}")

    with open(os.path.join(args.output, "build_volume_validation_results.csv"), "w", newline="") as f:
        all_fieldnames = []
        for r in results:
            for k in r.keys():
                if k not in all_fieldnames:
                    all_fieldnames.append(k)
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output}")
    print("No DICOM files were modified - read-only.")


if __name__ == "__main__":
    main()
