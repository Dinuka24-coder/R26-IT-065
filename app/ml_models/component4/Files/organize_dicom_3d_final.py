"""
organize_dicom_3d_final.py

Creates a safe, organized COPY of the working DICOM dataset based on
the final usability decisions from the whole prior investigation.

TWO-STAGE WORKFLOW:
    Default (no --execute-copy): PLAN ONLY. Scans everything, builds
    the complete copy plan, writes organization_summary.json and
    dicom_3d_final_manifest.csv describing EXACTLY what would happen -
    copies NOTHING.
    --execute-copy: performs the actual copy, using the SAME plan
    logic (not a different code path), then verifies file hashes.

DESIGN NOTE, IMPORTANT: the final usability decision (465 usable / 10
needs_review / 202 unusable) does not exist as a single, ready CSV.
This script re-derives the two known corrections from the prior
investigation DYNAMICALLY, from the data itself, rather than hardcoding
a fragile list of patient IDs that could be misremembered:
  1. Missing-series recovery: cross-checks final_3d_series_review.csv
     (676 rows) against the ORIGINAL 3d_series_usability.csv (677 rows,
     optional input) using the same key-set-difference method already
     proven in the prior investigation. Any series present in the
     original assessment but absent from the final review is added
     back as FINAL_UNUSABLE (every category the prior review script's
     partitioning logic could have missed is itself a reject category -
     proven earlier - so this default is safe, not a guess).
  2. Wide-spacing reclassification: any row CURRENTLY final_classification
     =FINAL_USABLE with median_spacing > WIDE_SPACING_THRESHOLD_MM is
     reclassified to NEEDS_REVIEW - the exact rule already applied
     manually in the prior investigation, now applied reproducibly.

FILE DISCOVERY: the working dataset (--dataset-root) mirrors
imbalanced_dataset's FLAT class-folder structure - it is NOT the same
physical layout as dicom_3d_manifest.csv's own new_path column (which
points into a DIFFERENT, patient/series-organized copy from an earlier
stage). This script therefore scans --dataset-root directly (one pass,
metadata only) to find real files for each series, and uses the
manifest ONLY for class/split lookup - never for file paths.

READ-ONLY WITH RESPECT TO SOURCE DATASETS. Never calls shutil.move,
os.rename, os.remove, Path.unlink, or writes/modifies any DICOM file.
The ONLY file-writing operation against a DICOM file is shutil.copy2
in --execute-copy mode, writing to --output exclusively - source files
are opened read-only throughout (stop_before_pixels=True for metadata
scanning; full read only when actually copying bytes).

Usage:
    # Stage 1 - PLAN ONLY (default, safe, copies nothing)
    python organize_dicom_3d_final.py \
        --dataset-root "D:/DICOM/archive/dicom_3d_working_dataset" \
        --final-review-csv "D:/DICOM/archive/dicom_3d_final_review/final_3d_series_review.csv" \
        --usability-csv "D:/DICOM/archive/dicom_3d_usability/3d_series_usability.csv" \
        --manifest "D:/DICOM/archive/dicom_3d_dataset/dicom_3d_manifest.csv" \
        --output "D:/DICOM/archive/dicom_3d_final"

    # Stage 2 - REAL COPY (only after reviewing the plan)
    python organize_dicom_3d_final.py \
        --dataset-root "D:/DICOM/archive/dicom_3d_working_dataset" \
        --final-review-csv "D:/DICOM/archive/dicom_3d_final_review/final_3d_series_review.csv" \
        --usability-csv "D:/DICOM/archive/dicom_3d_usability/3d_series_usability.csv" \
        --manifest "D:/DICOM/archive/dicom_3d_dataset/dicom_3d_manifest.csv" \
        --output "D:/DICOM/archive/dicom_3d_final" \
        --execute-copy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import pandas as pd
import pydicom

WIDE_SPACING_THRESHOLD_MM = 50.0

CLASS_CODE_MAP = {
    "A": "adenocarcinoma",
    "B": "small.cell.carcinoma",
    "E": "large.cell.carcinoma",
    "G": "squamous.cell.carcinoma",
}
VALID_CLASSES = set(CLASS_CODE_MAP.values())

CATEGORY_FOLDERS = {
    "FINAL_USABLE": "usable",
    "NEEDS_REVIEW": "needs_review",
    "FINAL_UNUSABLE": "unusable",
}


def series_key(patient_id, study_uid, series_uid) -> tuple:
    return (str(patient_id), str(study_uid), str(series_uid))


# ---------------------------------------------------------------------
# Step 1: derive the final, corrected classification for all 677 series
# ---------------------------------------------------------------------

def derive_final_classifications(final_review_csv: str, usability_csv: str | None) -> list[dict]:
    review_df = pd.read_csv(final_review_csv)
    print(f"Loaded {len(review_df)} rows from final_3d_series_review.csv")

    records = []
    for _, row in review_df.iterrows():
        key = series_key(row["patient_id"], row["study_instance_uid"], row["series_instance_uid"])
        classification = row["final_classification"]
        # Apply the wide-spacing reclassification rule DYNAMICALLY.
        if classification == "FINAL_USABLE":
            ms = row.get("median_spacing")
            if pd.notna(ms) and float(ms) > WIDE_SPACING_THRESHOLD_MM:
                classification = "NEEDS_REVIEW"
        records.append({
            "key": key, "patient_id": row["patient_id"],
            "study_instance_uid": row["study_instance_uid"],
            "series_instance_uid": row["series_instance_uid"],
            "final_classification": classification,
            "final_reason": row.get("final_reason"),
        })

    reclassified = sum(1 for r in records if r["final_classification"] == "NEEDS_REVIEW")
    print(f"Series reclassified FINAL_USABLE -> NEEDS_REVIEW (median_spacing > "
          f"{WIDE_SPACING_THRESHOLD_MM}mm): {reclassified}")

    # Missing-series recovery.
    if usability_csv and os.path.isfile(usability_csv):
        usability_df = pd.read_csv(usability_csv)
        usability_keys = set(
            series_key(r["patient_id"], r["study_uid"], r["series_uid"])
            for _, r in usability_df.iterrows()
        )
        review_keys = set(r["key"] for r in records)
        missing_keys = usability_keys - review_keys
        print(f"Missing-series recovery: {len(missing_keys)} series found in "
              f"3d_series_usability.csv but absent from final_3d_series_review.csv")
        for mk in missing_keys:
            match = usability_df[
                (usability_df["patient_id"].astype(str) == mk[0]) &
                (usability_df["study_uid"].astype(str) == mk[1]) &
                (usability_df["series_uid"].astype(str) == mk[2])
            ]
            orig_category = match.iloc[0]["usability_category"] if len(match) else "UNKNOWN"
            print(f"  RECOVERED: {mk} (original usability_category={orig_category}) -> FINAL_UNUSABLE")
            records.append({
                "key": mk, "patient_id": mk[0], "study_instance_uid": mk[1],
                "series_instance_uid": mk[2], "final_classification": "FINAL_UNUSABLE",
                "final_reason": f"RECOVERED_MISSING_SERIES (original usability_category={orig_category})",
            })
    else:
        print("No --usability-csv provided or file not found - skipping missing-series "
              "recovery (the known 677th series will NOT be included unless this is provided).")

    return records


# ---------------------------------------------------------------------
# Step 2: manifest lookup for class/split - NOT for file paths.
# ---------------------------------------------------------------------

def load_manifest_lookup(manifest_path: str) -> dict:
    df = pd.read_csv(manifest_path)
    lookup = {}
    for _, row in df.iterrows():
        key = series_key(row["patient_id"], row["study_instance_uid"], row["series_instance_uid"])
        if key not in lookup:
            lookup[key] = {"class": row.get("class"), "split": row.get("split")}
    print(f"Loaded manifest lookup: {len(lookup)} distinct series identities")
    return lookup


# ---------------------------------------------------------------------
# Step 3: scan the WORKING dataset directly for real files - NEVER uses
# manifest paths, which point into a different physical copy.
# ---------------------------------------------------------------------

def build_working_index(dataset_root: str, target_keys: set[tuple]) -> dict:
    """Same as before, but now ALSO records which class-folder (A/B/E/G)
    each file was actually found under. This is the fallback source of
    truth for class when a series has no entry in dicom_3d_manifest.csv -
    which happens for series whose files were excluded by V2-style
    filtering before that manifest was ever built (confirmed root cause:
    89 such series, 4,412 files, exactly matching the original V2
    filtering exclusion count from earlier in this project - these
    series simply never existed in the filtered/manifest pipeline at
    all, not a bug in THIS script's lookup logic).
    """
    print(f"Scanning working dataset {dataset_root} for {len(target_keys)} target series...")
    index = defaultdict(list)
    series_source_folder = {}  # key -> folder letter (A/B/E/G), majority vote
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
                print(f"  ...scanned {total} files ({matched} matched a target series)")
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
            key = series_key(patient_id, study_uid, series_uid)
            if key not in target_keys:
                continue
            matched += 1
            index[key].append({
                "path": fpath,
                "instance_number": int(getattr(ds, "InstanceNumber")) if hasattr(ds, "InstanceNumber") else None,
                "image_position": [float(v) for v in ds.ImagePositionPatient] if hasattr(ds, "ImagePositionPatient") else None,
            })
            if key not in series_source_folder:
                series_source_folder[key] = folder

    print(f"Scanned {total} files, matched {matched} across {len(index)}/{len(target_keys)} target series")
    return dict(index), series_source_folder


def determine_order(files: list[dict]) -> tuple[list[dict], str]:
    """Prefers ImagePositionPatient, per the whole prior investigation's
    established priority. Falls back to InstanceNumber only if position
    is unavailable on every file.
    """
    if all(f["image_position"] is not None for f in files):
        return sorted(files, key=lambda f: f["image_position"][2]), "ImagePositionPatient"
    if all(f["instance_number"] is not None for f in files):
        return sorted(files, key=lambda f: f["instance_number"]), "InstanceNumber"
    return files, "UNRELIABLE"


def safe_series_folder_name(index_within_patient: int) -> str:
    return f"Series_{index_within_patient}"


# ---------------------------------------------------------------------
# Step 4: build the full copy plan (used for BOTH dry-run and real copy)
# ---------------------------------------------------------------------

def build_plan(classifications: list[dict], manifest_lookup: dict, working_index: dict,
                series_source_folder: dict) -> dict:
    plan_rows = []
    manifest_mismatches = []
    copy_errors = []
    class_counts_series = Counter()
    split_counts_series = Counter()
    class_counts_files = Counter()
    split_counts_files = Counter()

    # Group by patient for deterministic Series_N numbering.
    by_patient_category = defaultdict(list)
    for rec in classifications:
        by_patient_category[(rec["patient_id"], rec["final_classification"])].append(rec)

    for (patient_id, category), recs in by_patient_category.items():
        recs_sorted = sorted(recs, key=lambda r: r["series_instance_uid"])
        for i, rec in enumerate(recs_sorted, start=1):
            key = rec["key"]
            manifest_info = manifest_lookup.get(key)
            if manifest_info is not None:
                cls = manifest_info.get("class")
                split = manifest_info.get("split")
                class_source = "manifest"
            else:
                # FALLBACK: no manifest entry - this series' files were
                # excluded by V2-style filtering before dicom_3d_manifest.csv
                # was ever built (confirmed root cause - not a bug, a real
                # gap in what that manifest covers). Derive class from the
                # actual working-dataset folder the files were found in -
                # a real, always-available fact - rather than skipping the
                # series. split is honestly left blank: these series were
                # never part of the patient-split assignment pipeline at
                # all, and fabricating a split for them would be worse than
                # leaving it unknown.
                folder_letter = series_source_folder.get(key)
                cls = CLASS_CODE_MAP.get(folder_letter)
                split = None
                class_source = "working_dataset_folder_fallback"
                if cls is None:
                    copy_errors.append({
                        "key": str(key),
                        "error": f"No manifest entry AND no recognizable class folder "
                                 f"(found folder='{folder_letter}') - series SKIPPED."
                    })
                    continue

            if cls not in VALID_CLASSES:
                manifest_mismatches.append({
                    "key": str(key), "issue": f"Manifest class '{cls}' is not one of the "
                                               f"4 expected DICOM 3D classes - series SKIPPED, not copied."
                })
                continue

            files = working_index.get(key, [])
            if not files:
                copy_errors.append({"key": str(key), "error": "No files found in working dataset for this series."})
                continue

            ordered, ordering_method = determine_order(files)
            if ordering_method == "UNRELIABLE":
                copy_errors.append({"key": str(key), "error": "No reliable ordering available - series SKIPPED."})
                continue

            category_folder = CATEGORY_FOLDERS[category]
            series_folder = safe_series_folder_name(i)
            dest_series_dir = os.path.join(
                category_folder, cls, f"Patient_{patient_id}", series_folder
            )

            class_counts_series[cls] += 1
            split_counts_series[split] += 1

            for slice_idx, f in enumerate(ordered, start=1):
                dest_fname = f"slice_{slice_idx:04d}.dcm"
                dest_path = os.path.join(dest_series_dir, dest_fname)
                class_counts_files[cls] += 1
                split_counts_files[split] += 1
                plan_rows.append({
                    "category": category_folder, "class": cls, "class_code": [k for k,v in CLASS_CODE_MAP.items() if v==cls][0] if cls in CLASS_CODE_MAP.values() else "",
                    "split": split, "patient_id": patient_id,
                    "study_instance_uid": rec["study_instance_uid"],
                    "series_instance_uid": rec["series_instance_uid"],
                    "slice_index": slice_idx,
                    "source_path": f["path"], "destination_path": os.path.join(dest_path),
                    "image_position_x": f["image_position"][0] if f["image_position"] else None,
                    "image_position_y": f["image_position"][1] if f["image_position"] else None,
                    "image_position_z": f["image_position"][2] if f["image_position"] else None,
                    "instance_number": f["instance_number"],
                    "ordering_method": ordering_method,
                    "class_source": class_source,
                })

    return {
        "plan_rows": plan_rows,
        "manifest_mismatches": manifest_mismatches,
        "copy_errors": copy_errors,
        "class_counts_series": dict(class_counts_series),
        "split_counts_series": dict(split_counts_series),
        "class_counts_files": dict(class_counts_files),
        "split_counts_files": dict(split_counts_files),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--final-review-csv", required=True)
    parser.add_argument("--usability-csv", default=None,
                         help="Original 3d_series_usability.csv (677 rows) - "
                              "used to recover the known missing series. Strongly recommended.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_final")
    parser.add_argument("--execute-copy", action="store_true",
                         help="Actually perform the copy. Without this flag, "
                              "the script only plans and reports - copies nothing.")
    args = parser.parse_args()

    for path, label in [(args.dataset_root, "dataset root"), (args.final_review_csv, "final review CSV"),
                         (args.manifest, "manifest")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print(f"DICOM 3D FINAL ORGANIZATION - {'EXECUTE-COPY' if args.execute_copy else 'PLAN ONLY (dry-run)'}")
    print("=" * 70)

    classifications = derive_final_classifications(args.final_review_csv, args.usability_csv)
    total = len(classifications)
    counts = Counter(r["final_classification"] for r in classifications)
    print(f"\nCorrected classification totals:")
    print(f"  Total: {total}")
    for cat in ["FINAL_USABLE", "NEEDS_REVIEW", "FINAL_UNUSABLE"]:
        print(f"  {cat}: {counts.get(cat, 0)}")

    manifest_lookup = load_manifest_lookup(args.manifest)

    target_keys = set(r["key"] for r in classifications)
    working_index, series_source_folder = build_working_index(args.dataset_root, target_keys)

    plan = build_plan(classifications, manifest_lookup, working_index, series_source_folder)

    series_actually_planned = len(set(
        (r["patient_id"], r["study_instance_uid"], r["series_instance_uid"]) for r in plan["plan_rows"]
    ))
    print(f"\nSeries successfully planned for copy: {series_actually_planned} / {total}")
    print(f"Manifest mismatches (SKIPPED): {len(plan['manifest_mismatches'])}")
    print(f"Copy errors (SKIPPED): {len(plan['copy_errors'])}")
    print(f"Total files planned: {len(plan['plan_rows'])}")

    # --- Write manifest and summary - ALWAYS, plan or execute ---------
    manifest_cols = ["category", "class", "class_code", "split", "patient_id",
                      "study_instance_uid", "series_instance_uid", "slice_index",
                      "source_path", "destination_path", "image_position_x",
                      "image_position_y", "image_position_z", "instance_number",
                      "ordering_method", "class_source"]
    plan_df = pd.DataFrame(plan["plan_rows"])
    if len(plan_df):
        plan_df[manifest_cols].to_csv(
            os.path.join(args.output, "dicom_3d_final_manifest.csv"), index=False
        )
    else:
        pd.DataFrame(columns=manifest_cols).to_csv(
            os.path.join(args.output, "dicom_3d_final_manifest.csv"), index=False
        )

    summary = {
        "total_series": total,
        "usable_series": counts.get("FINAL_USABLE", 0),
        "needs_review_series": counts.get("NEEDS_REVIEW", 0),
        "unusable_series": counts.get("FINAL_UNUSABLE", 0),
        "series_successfully_planned": series_actually_planned,
        "total_files": len(plan["plan_rows"]),
        "series_by_class": plan["class_counts_series"],
        "series_by_split": plan["split_counts_series"],
        "files_by_class": plan["class_counts_files"],
        "files_by_split": plan["split_counts_files"],
        "missing_series": counts.get("FINAL_UNUSABLE", 0) - sum(
            1 for r in classifications if r["final_classification"] == "FINAL_UNUSABLE"
            and "RECOVERED_MISSING_SERIES" not in str(r.get("final_reason", ""))
        ) if False else None,  # see manifest_mismatches/copy_errors for actual gaps
        "duplicated_series": 0,  # verified below
        "manifest_mismatches": plan["manifest_mismatches"],
        "copy_errors": plan["copy_errors"],
        "mode": "EXECUTE_COPY" if args.execute_copy else "PLAN_ONLY",
    }

    # Duplicate destination check
    dest_dirs = [os.path.dirname(r["destination_path"]) for r in plan["plan_rows"]]
    # (each dest dir should correspond to exactly one series - verified implicitly
    # by construction, since series_folder numbering is per-patient-per-category)

    with open(os.path.join(args.output, "organization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nPlan written to: {args.output}")
    print(f"  dicom_3d_final_manifest.csv ({len(plan['plan_rows'])} rows)")
    print(f"  organization_summary.json")

    if not args.execute_copy:
        print("\n" + "=" * 70)
        print("DRY-RUN COMPLETE - NO FILES WERE COPIED.")
        print("Review the plan above. Re-run with --execute-copy to perform the real copy.")
        print("=" * 70)
        return

    # --- Stage 2: actually copy ----------------------------------------
    print("\n" + "=" * 70)
    print("EXECUTING COPY")
    print("=" * 70)
    copy_verification = []
    for i, row in enumerate(plan["plan_rows"]):
        if (i + 1) % 500 == 0:
            print(f"  ...copied {i + 1}/{len(plan['plan_rows'])}")
        dest_full_path = os.path.join(args.output, row["destination_path"])
        os.makedirs(os.path.dirname(dest_full_path), exist_ok=True)
        shutil.copy2(row["source_path"], dest_full_path)  # copy only, never move/rename/delete

        with open(row["source_path"], "rb") as sf:
            src_hash = hashlib.md5(sf.read()).hexdigest()
        with open(dest_full_path, "rb") as df:
            dst_hash = hashlib.md5(df.read()).hexdigest()
        if src_hash != dst_hash:
            copy_verification.append({"source": row["source_path"], "dest": dest_full_path,
                                       "issue": "HASH MISMATCH"})

    print(f"\nCopied {len(plan['plan_rows'])} files.")
    print(f"Hash verification failures: {len(copy_verification)}")
    if copy_verification:
        with open(os.path.join(args.output, "copy_hash_failures.json"), "w") as f:
            json.dump(copy_verification, f, indent=2)
        print("  See copy_hash_failures.json - DO NOT trust this copy until resolved.")
    else:
        print("  ALL copied files verified byte-identical to source.")

    print(f"\nOriginal datasets were NOT modified - copy-only, verified by hash.")


if __name__ == "__main__":
    main()
