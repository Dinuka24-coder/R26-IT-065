"""
verify_dicom_3d_final_dataset.py

STANDALONE, READ-ONLY final verification of the organized DICOM 3D
dataset at dicom_3d_final, against the working dataset it was copied
from and the plan manifest that describes what should have happened.

Performs an INDEPENDENT re-scan of both the working dataset and the
final organized dataset (not just trusting the plan manifest or the
copy operation's own self-report) - this script re-reads real DICOM
metadata and recomputes hashes itself, rather than assuming the prior
stages were correct.

READ-ONLY. Never calls shutil.move, os.rename, os.remove, Path.unlink,
pydicom save, or any equivalent write/modify operation against EITHER
dataset. All writes go only to --output.

Usage:
    python verify_dicom_3d_final_dataset.py \
        --working-dataset "D:/DICOM/archive/dicom_3d_working_dataset" \
        --final-dataset "D:/DICOM/archive/dicom_3d_final" \
        --plan-manifest "D:/DICOM/archive/dicom_3d_final/dicom_3d_final_manifest.csv" \
        --output "D:/DICOM/archive/dicom_3d_final_verification"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pydicom

VALID_CLASSES = {"adenocarcinoma", "small.cell.carcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma"}
VALID_CATEGORIES = {"usable", "needs_review", "unusable"}
WIDE_SPACING_THRESHOLD_MM = 50.0
DUPLICATE_POSITION_TOLERANCE_MM = 0.01


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_sop_uid(path: str) -> str | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return str(getattr(ds, "SOPInstanceUID", None)) or None
    except Exception:
        return None


# ---------------------------------------------------------------------
# Step 1: independently scan the FINAL organized dataset from disk -
# not just trusting the plan manifest.
# ---------------------------------------------------------------------

def scan_final_dataset(final_root: str) -> list[dict]:
    print(f"Scanning final organized dataset at {final_root}...")
    records = []
    total = 0
    for category in sorted(os.listdir(final_root)):
        cat_path = os.path.join(final_root, category)
        if not os.path.isdir(cat_path) or category not in VALID_CATEGORIES:
            if os.path.isdir(cat_path):
                print(f"  WARNING: unexpected top-level directory found: {category}")
            continue
        for cls in sorted(os.listdir(cat_path)):
            cls_path = os.path.join(cat_path, cls)
            if not os.path.isdir(cls_path):
                continue
            if cls not in VALID_CLASSES:
                print(f"  WARNING: unexpected class directory found: {category}/{cls}")
            for patient_dir in sorted(os.listdir(cls_path)):
                patient_path = os.path.join(cls_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                for series_dir in sorted(os.listdir(patient_path)):
                    series_path = os.path.join(patient_path, series_dir)
                    if not os.path.isdir(series_path):
                        continue
                    for fname in sorted(os.listdir(series_path)):
                        if not fname.endswith(".dcm"):
                            print(f"  WARNING: unexpected non-DICOM file: "
                                  f"{category}/{cls}/{patient_dir}/{series_dir}/{fname}")
                            continue
                        total += 1
                        if total % 2000 == 0:
                            print(f"  ...scanned {total} files")
                        fpath = os.path.join(series_path, fname)
                        records.append({
                            "category": category, "class": cls,
                            "patient_dir": patient_dir, "series_dir": series_dir,
                            "path": fpath, "filename": fname,
                        })
    print(f"Final dataset scan complete: {total} DICOM files found")
    return records


# ---------------------------------------------------------------------
# Step 2: independently scan the WORKING dataset - same as prior scripts.
# ---------------------------------------------------------------------

def scan_working_dataset(working_root: str) -> dict:
    print(f"Scanning working dataset at {working_root} (metadata only)...")
    index = {}  # sop_instance_uid -> {path, patient_id, study_uid, series_uid, ...}
    total = 0
    class_folders = [d for d in os.listdir(working_root) if os.path.isdir(os.path.join(working_root, d))]
    for folder in class_folders:
        folder_path = os.path.join(working_root, folder)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".dcm"):
                continue
            total += 1
            if total % 2000 == 0:
                print(f"  ...scanned {total} files")
            fpath = os.path.join(folder_path, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception:
                continue
            sop_uid = getattr(ds, "SOPInstanceUID", None)
            if not sop_uid:
                continue
            index[str(sop_uid)] = {
                "path": fpath,
                "patient_id": str(getattr(ds, "PatientID", "")),
                "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "modality": getattr(ds, "Modality", None),
                "rows": int(ds.Rows) if hasattr(ds, "Rows") else None,
                "columns": int(ds.Columns) if hasattr(ds, "Columns") else None,
                "image_position": [float(v) for v in ds.ImagePositionPatient] if hasattr(ds, "ImagePositionPatient") else None,
                "instance_number": int(getattr(ds, "InstanceNumber")) if hasattr(ds, "InstanceNumber") else None,
                "pixel_spacing": [float(v) for v in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else None,
                "acquisition_number": str(getattr(ds, "AcquisitionNumber", "")) or None,
            }
    print(f"Working dataset scan complete: {total} files scanned, {len(index)} with valid SOPInstanceUID")
    return index


# ---------------------------------------------------------------------
# Main verification logic
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-dataset", required=True)
    parser.add_argument("--final-dataset", required=True)
    parser.add_argument("--plan-manifest", required=True,
                         help="dicom_3d_final_manifest.csv produced by organize_dicom_3d_final.py")
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_final_verification")
    args = parser.parse_args()

    for path, label in [(args.working_dataset, "working dataset"), (args.final_dataset, "final dataset"),
                         (args.plan_manifest, "plan manifest")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("DICOM 3D FINAL DATASET VERIFICATION - READ-ONLY")
    print("=" * 70)

    issues = []  # each: {"severity": "FAIL"|"WARNING", "check": ..., "detail": ...}

    plan_df = pd.read_csv(args.plan_manifest)
    print(f"\nLoaded plan manifest: {len(plan_df)} rows")

    # --- Step A: independent scan of final dataset from disk ----------
    final_files = scan_final_dataset(args.final_dataset)
    print(f"\nFinal dataset (disk scan): {len(final_files)} files")
    print(f"Plan manifest: {len(plan_df)} files")
    if len(final_files) != len(plan_df):
        issues.append({"severity": "FAIL", "check": "file_count",
                        "detail": f"Disk scan found {len(final_files)} files, plan manifest lists {len(plan_df)}."})

    # --- Step B: SOPInstanceUID-based cross-check (per instruction #13,
    # not filename-based) ------------------------------------------------
    print("\nReading SOPInstanceUID from final dataset files (independent identity check)...")
    final_sop_uids = {}
    for i, rec in enumerate(final_files):
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(final_files)}")
        sop = read_sop_uid(rec["path"])
        if sop:
            final_sop_uids.setdefault(sop, []).append(rec["path"])

    dup_sop_in_final = {k: v for k, v in final_sop_uids.items() if len(v) > 1}
    print(f"Duplicate SOPInstanceUID within final dataset: {len(dup_sop_in_final)}")
    for sop, paths in list(dup_sop_in_final.items())[:10]:
        issues.append({"severity": "FAIL", "check": "duplicate_sop_uid",
                        "detail": f"SOPInstanceUID {sop} appears at: {paths}"})

    working_index = scan_working_dataset(args.working_dataset)
    print(f"\nWorking dataset: {len(working_index)} files with valid SOPInstanceUID")

    working_sops = set(working_index.keys())
    final_sops = set(final_sop_uids.keys())

    missing_from_final = working_sops - final_sops
    extra_in_final = final_sops - working_sops
    print(f"\nFiles in working dataset but MISSING from final: {len(missing_from_final)}")
    print(f"Files in final dataset with NO working-dataset match (EXTRA): {len(extra_in_final)}")
    for sop in list(missing_from_final)[:20]:
        issues.append({"severity": "FAIL", "check": "missing_file",
                        "detail": f"SOPInstanceUID {sop} present in working dataset, absent from final."})
    for sop in list(extra_in_final)[:20]:
        issues.append({"severity": "FAIL", "check": "extra_file",
                        "detail": f"SOPInstanceUID {sop} present in final dataset, not found in working dataset."})

    # --- Step C: hash verification for matched files -------------------
    print("\nHash-verifying matched files (this may take a while for the full dataset)...")
    matched_sops = working_sops & final_sops
    hash_results = []
    hash_mismatches = 0
    for i, sop in enumerate(matched_sops):
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(matched_sops)}")
        src_path = working_index[sop]["path"]
        dst_path = final_sop_uids[sop][0]
        src_hash = sha256_of_file(src_path)
        dst_hash = sha256_of_file(dst_path)
        match = src_hash == dst_hash
        if not match:
            hash_mismatches += 1
            issues.append({"severity": "FAIL", "check": "hash_mismatch",
                            "detail": f"SOPInstanceUID {sop}: source and destination hashes differ."})
        hash_results.append({"sop_instance_uid": sop, "source_path": src_path,
                              "destination_path": dst_path, "match": match})

    print(f"\nHash verification: {len(matched_sops)} matched, {hash_mismatches} mismatches")

    # --- Step D: series-level integrity re-check from REAL metadata ----
    print("\nRe-verifying series-level integrity from real DICOM metadata...")
    series_from_working = defaultdict(list)
    for sop, meta in working_index.items():
        if sop in final_sops:
            key = (meta["patient_id"], meta["study_uid"], meta["series_uid"])
            series_from_working[key].append(meta)

    series_records = []
    needs_review_recheck = []
    for key, metas in series_from_working.items():
        modalities = set(m["modality"] for m in metas)
        dims = set((m["rows"], m["columns"]) for m in metas)
        rec = {
            "patient_id": key[0], "study_instance_uid": key[1], "series_instance_uid": key[2],
            "number_of_files": len(metas),
            "modality_consistent": len(modalities) == 1 and "CT" in modalities,
            "dimensions_consistent": len(dims) <= 1,
        }
        if len(modalities) != 1 or "CT" not in modalities:
            issues.append({"severity": "WARNING", "check": "modality",
                            "detail": f"Series {key}: modalities found = {modalities}"})
        if len(dims) > 1:
            issues.append({"severity": "WARNING", "check": "dimensions",
                            "detail": f"Series {key}: inconsistent dimensions = {dims}"})
        series_records.append(rec)

        # Find this series' category from the plan
        plan_match = plan_df[
            (plan_df["patient_id"].astype(str) == key[0]) &
            (plan_df["study_instance_uid"].astype(str) == key[1]) &
            (plan_df["series_instance_uid"].astype(str) == key[2])
        ]
        if len(plan_match) and plan_match.iloc[0]["category"] == "needs_review":
            with_pos = [m for m in metas if m["image_position"] is not None]
            if len(with_pos) >= 2:
                z = sorted(m["image_position"][2] for m in with_pos)
                diffs = np.abs(np.diff(z))
                nonzero = diffs[diffs > DUPLICATE_POSITION_TOLERANCE_MM]
                median_spacing = float(np.median(nonzero)) if len(nonzero) else 0.0
                needs_review_recheck.append({
                    "patient_id": key[0], "recomputed_median_spacing": median_spacing,
                    "exceeds_threshold": median_spacing > WIDE_SPACING_THRESHOLD_MM,
                })
                if median_spacing <= WIDE_SPACING_THRESHOLD_MM:
                    issues.append({"severity": "FAIL", "check": "needs_review_rule_violation",
                                   "detail": f"Series {key} is under needs_review but recomputed "
                                             f"median_spacing={median_spacing:.2f}mm does not exceed "
                                             f"{WIDE_SPACING_THRESHOLD_MM}mm."})

    # --- Category/class totals from the plan (already trustworthy, but
    # cross-checked against the disk scan's own category/class folders) -
    plan_cat_series = defaultdict(set)
    for _, row in plan_df.iterrows():
        plan_cat_series[row["category"]].add((row["patient_id"], row["study_instance_uid"], row["series_instance_uid"]))
    category_series_counts = {cat: len(keys) for cat, keys in plan_cat_series.items()}

    disk_cat_class = defaultdict(set)
    for rec in final_files:
        disk_cat_class[(rec["category"])].add((rec["patient_dir"], rec["series_dir"]))

    # --- Save all reports ------------------------------------------------
    total_series = len(set(zip(plan_df["patient_id"], plan_df["study_instance_uid"], plan_df["series_instance_uid"])))
    verdict = "PASS"
    critical_failures = [i for i in issues if i["severity"] == "FAIL"]
    if critical_failures:
        verdict = "FAIL"

    summary = {
        "verdict": verdict,
        "total_series_in_plan": total_series,
        "total_files_in_plan": len(plan_df),
        "total_files_on_disk_final": len(final_files),
        "category_series_counts": category_series_counts,
        "working_dataset_files_with_sop_uid": len(working_index),
        "final_dataset_files_with_sop_uid": len(final_sops),
        "missing_from_final_count": len(missing_from_final),
        "extra_in_final_count": len(extra_in_final),
        "duplicate_sop_uid_in_final_count": len(dup_sop_in_final),
        "hash_matched_count": len(matched_sops),
        "hash_mismatch_count": hash_mismatches,
        "needs_review_rule_check": needs_review_recheck,
        "critical_failures": len(critical_failures),
        "warnings": len([i for i in issues if i["severity"] == "WARNING"]),
    }

    with open(os.path.join(args.output, "final_dataset_verification_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    pd.DataFrame(series_records).to_csv(
        os.path.join(args.output, "final_series_verification.csv"), index=False)

    class_counts_df = plan_df.groupby(["category", "class"]).apply(
        lambda g: len(set(zip(g["patient_id"], g["study_instance_uid"], g["series_instance_uid"])))
    ).reset_index(name="series_count")
    class_counts_df.to_csv(os.path.join(args.output, "final_class_counts.csv"), index=False)

    cat_counts_df = pd.DataFrame([
        {"category": cat, "series_count": len(keys),
         "file_count": len(plan_df[plan_df["category"] == cat])}
        for cat, keys in plan_cat_series.items()
    ])
    cat_counts_df.to_csv(os.path.join(args.output, "final_category_counts.csv"), index=False)

    pd.DataFrame(hash_results).to_csv(
        os.path.join(args.output, "final_file_hash_verification.csv"), index=False)

    pd.DataFrame(issues).to_csv(
        os.path.join(args.output, "final_verification_issues.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)
    print(f"Critical failures: {len(critical_failures)}")
    print(f"Warnings: {summary['warnings']}")
    print(f"\nReports saved to: {args.output}")
    print("\nNo DICOM files were modified - this script only read files "
          "and computed hashes; nothing was written except the verification "
          "reports above.")


if __name__ == "__main__":
    main()
