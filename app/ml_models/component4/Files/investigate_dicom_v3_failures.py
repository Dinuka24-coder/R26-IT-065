"""
investigate_dicom_v3_failures.py

STANDALONE, READ-ONLY investigation script. Determines whether the
2.55% build_volume() pass rate is primarily caused by the V2-style
filtering/organization process (missing interior slices, multiple
acquisitions merged under one SeriesInstanceUID) or by genuinely
unsuitable source DICOM data - by tracing failed organized series back
to the ORIGINAL 18,500-file source dataset and examining exactly which
files were excluded and why.

Does NOT call build_volume(), does NOT import anything from app.* -
this is pure DICOM metadata analysis using only pydicom/pandas/numpy,
so there is no project sys.path/import concern the way there was for
scripts that reuse production inference code.

STRICTLY READ-ONLY with respect to both datasets. Never calls
shutil.copy/move, os.rename, os.remove, Path.rename/unlink, or any
equivalent write/delete operation against imbalanced_dataset or
dicom_3d_dataset. The only thing this script writes is a NEW,
separate output directory containing investigation reports.

V2-style filtering rules (reused exactly, not reinterpreted):
    included  = Modality == "CT" AND RescaleSlope present
                AND RescaleIntercept present AND PatientID present
    excluded reasons (a file can have more than one, all reported):
        non_ct              - Modality != "CT" (or missing)
        missing_rescale      - RescaleSlope or RescaleIntercept absent
        missing_patient_id   - PatientID absent/empty

Usage:
    python investigate_dicom_v3_failures.py \
        --dataset-root "D:/DICOM/archive/imbalanced_dataset" \
        --validation-report "D:/DICOM/archive/dicom_3d_dataset/v3_volume_validation_report.csv" \
        --manifest "D:/DICOM/archive/dicom_3d_dataset/dicom_3d_manifest.csv" \
        --output "D:/DICOM/archive/dicom_3d_investigation" \
        --sample-size 20

    or, for a complete run over every failure:
        ... --all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom

RANDOM_SEED = 42

# Gap-flagging threshold, stated explicitly: a gap between two
# consecutive INCLUDED slices is flagged as "unusually large" if it
# exceeds 2x the series' own median included-slice gap. Relative to
# each series' own typical spacing, not a fixed absolute mm value,
# since different series/protocols have different normal spacing.
GAP_RELATIVE_THRESHOLD = 2.0


# ---------------------------------------------------------------------
# Step 1: build an index of the ORIGINAL dataset - ONE pass, not
# rescanned per series.
# ---------------------------------------------------------------------

def build_original_index(dataset_root: str) -> dict:
    """Scans every .dcm file under dataset_root ONCE (metadata only,
    stop_before_pixels=True - no pixel decoding needed for this
    investigation), and returns a dict keyed by
    (patient_id, study_uid, series_uid) -> list of per-file metadata
    dicts. This is the efficient index the investigation queries
    against, rather than rescanning 18,500 files per series.
    """
    print(f"Building metadata index of {dataset_root} (one pass, "
          f"metadata only, no pixel decoding)...")
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
            except Exception as exc:
                unreadable += 1
                continue

            patient_id = getattr(ds, "PatientID", None)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            series_uid = getattr(ds, "SeriesInstanceUID", None)

            modality = getattr(ds, "Modality", None)
            has_rescale = hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept")
            rescale_slope = getattr(ds, "RescaleSlope", None)
            rescale_intercept = getattr(ds, "RescaleIntercept", None)
            instance_number = getattr(ds, "InstanceNumber", None)
            image_position = list(ds.ImagePositionPatient) if hasattr(ds, "ImagePositionPatient") else None

            reasons = []
            if modality != "CT":
                reasons.append("non_ct")
            if not has_rescale:
                reasons.append("missing_rescale")
            if not patient_id:
                reasons.append("missing_patient_id")
            included = len(reasons) == 0

            meta = {
                "path": fpath,
                "folder_class": folder,
                "patient_id": str(patient_id) if patient_id else None,
                "study_uid": str(study_uid) if study_uid else None,
                "series_uid": str(series_uid) if series_uid else None,
                "instance_number": int(instance_number) if instance_number is not None else None,
                "image_position": [float(v) for v in image_position] if image_position else None,
                "modality": modality,
                "rescale_slope": float(rescale_slope) if rescale_slope is not None else None,
                "rescale_intercept": float(rescale_intercept) if rescale_intercept is not None else None,
                "included": included,
                "exclusion_reasons": reasons,
            }

            if patient_id and study_uid and series_uid:
                key = (str(patient_id), str(study_uid), str(series_uid))
                index[key].append(meta)

    print(f"Indexed {total} files ({unreadable} unreadable). "
          f"{len(index)} distinct (patient, study, series) groups found.")
    return dict(index)


# ---------------------------------------------------------------------
# Step 2: load the validation report and select failures to investigate
# ---------------------------------------------------------------------

def load_failures(validation_report_path: str, category: str) -> list[dict]:
    df = pd.read_csv(validation_report_path)
    matches = df[df["failure_category"] == category]
    return matches.to_dict("records")


def select_series(failures: list[dict], sample_size: int | None, use_all: bool) -> list[dict]:
    if use_all or sample_size is None or sample_size >= len(failures):
        return failures
    rng = random.Random(RANDOM_SEED)
    return rng.sample(failures, sample_size)


# ---------------------------------------------------------------------
# Step 3-6: spacing gap investigation
# ---------------------------------------------------------------------

def investigate_spacing_series(key: tuple, original_files: list[dict]) -> dict:
    all_with_pos = [f for f in original_files if f["image_position"] is not None]
    all_sorted = sorted(all_with_pos, key=lambda f: f["image_position"][2])

    included = [f for f in all_sorted if f["included"]]
    excluded = [f for f in all_sorted if not f["included"]]

    result = {
        "patient_id": key[0], "study_uid": key[1], "series_uid": key[2],
        "original_file_count": len(original_files),
        "with_position_count": len(all_with_pos),
        "included_file_count": len(included),
        "excluded_file_count": len(excluded),
        "included_z": [f["image_position"][2] for f in included],
        "excluded_z": [f["image_position"][2] for f in excluded],
        "gaps": [],
    }

    if len(included) < 2:
        result["note"] = "Fewer than 2 included slices with position - cannot analyze gaps."
        return result

    z_included = [f["image_position"][2] for f in included]
    diffs = np.abs(np.diff(z_included))
    median_gap = float(np.median(diffs)) if len(diffs) else 0.0
    threshold = median_gap * GAP_RELATIVE_THRESHOLD

    for i in range(len(z_included) - 1):
        gap_size = float(abs(z_included[i + 1] - z_included[i]))
        if gap_size <= threshold or median_gap == 0:
            continue
        lo, hi = sorted([z_included[i], z_included[i + 1]])
        excluded_in_gap = [
            f for f in excluded
            if f["image_position"] is not None and lo < f["image_position"][2] < hi
        ]
        result["gaps"].append({
            "z_before": z_included[i], "z_after": z_included[i + 1],
            "gap_size_mm": gap_size,
            "median_gap_mm": median_gap,
            "gap_explained_by_excluded_slice": len(excluded_in_gap) > 0,
            "excluded_files_in_gap": [
                {
                    "path": f["path"], "z": f["image_position"][2],
                    "modality": f["modality"],
                    "rescale_slope": f["rescale_slope"],
                    "rescale_intercept": f["rescale_intercept"],
                    "exclusion_reasons": f["exclusion_reasons"],
                }
                for f in excluded_in_gap
            ],
        })

    return result


# ---------------------------------------------------------------------
# Step 7-8: non-monotonic / multi-sweep investigation
# ---------------------------------------------------------------------

def count_monotonic_runs(z: list[float]) -> tuple[int, list[int]]:
    if len(z) < 2:
        return 1, [len(z)]
    diffs = np.diff(z)
    signs = np.sign(diffs)
    runs = []
    current_len = 1
    current_sign = signs[0] if signs[0] != 0 else 1
    for s in signs[1:]:
        if s == current_sign or s == 0:
            current_len += 1
        else:
            runs.append(current_len)
            current_len = 1
            current_sign = s
    runs.append(current_len)
    return len(runs), runs


def investigate_nonmonotonic_series(key: tuple, original_files: list[dict]) -> dict:
    with_both = [f for f in original_files
                 if f["image_position"] is not None and f["instance_number"] is not None]
    ordered_by_instance = sorted(with_both, key=lambda f: f["instance_number"])

    included = [f for f in original_files if f["included"]]
    excluded = [f for f in original_files if not f["included"]]

    result = {
        "patient_id": key[0], "study_uid": key[1], "series_uid": key[2],
        "original_file_count": len(original_files),
        "included_file_count": len(included),
        "excluded_file_count": len(excluded),
        "num_ordered_with_position": len(ordered_by_instance),
    }

    if len(ordered_by_instance) < 3:
        result["conclusion"] = "UNDETERMINED"
        result["evidence"] = "Fewer than 3 slices with both InstanceNumber and ImagePositionPatient."
        return result

    z_seq = [f["image_position"][2] for f in ordered_by_instance]
    n_runs, run_lengths = count_monotonic_runs(z_seq)

    # Identify "sweeps" as the long runs (excluding trivial length-1
    # transition points), and report their z-ranges.
    sweeps = []
    idx = 0
    for length in run_lengths:
        segment = z_seq[idx: idx + length + 1] if idx == 0 else z_seq[idx - 1: idx + length]
        if length >= 3:  # ignore trivial 1-2 length "runs" (mere transition points)
            sweeps.append({
                "length": length,
                "z_start": segment[0] if segment else None,
                "z_end": segment[-1] if segment else None,
            })
        idx += length

    long_runs = [r for r in run_lengths if r >= 3]

    if len(long_runs) >= 2:
        lengths_arr = np.array(long_runs)
        similar_lengths = (lengths_arr.max() - lengths_arr.min()) <= max(5, 0.3 * lengths_arr.mean())
        confidence = "strong" if (len(long_runs) >= 2 and similar_lengths) else "moderate"
        result["conclusion"] = "LIKELY MULTIPLE ACQUISITIONS"
        result["confidence"] = confidence
        result["evidence"] = {
            "num_long_monotonic_runs": len(long_runs),
            "run_lengths": long_runs,
            "sweeps": sweeps,
            "reasoning": (
                f"{len(long_runs)} separate long monotonic z-runs found "
                f"within one nominal series identity, with lengths "
                f"{long_runs} - " +
                ("similar in length, consistent with repeated/duplicated "
                 "acquisitions of the same anatomy."
                 if similar_lengths else
                 "differing in length, weaker evidence for a clean repeat "
                 "pattern, but still multiple distinct sweeps.")
            ),
        }
    elif n_runs == 1:
        result["conclusion"] = "UNDETERMINED"
        result["evidence"] = (
            "Sequence appears monotonic in this analysis despite being "
            "flagged by build_volume() - likely an exact-tie z-position "
            "(build_volume() uses strict inequality) rather than multiple "
            "acquisitions."
        )
    else:
        result["conclusion"] = "UNDETERMINED"
        result["evidence"] = (
            f"{n_runs} runs found (lengths {run_lengths}) but insufficient "
            f"long runs to confidently indicate multiple acquisitions "
            f"rather than a few individually out-of-place slices."
        )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                         help="Original source dataset, e.g. "
                              "D:\\DICOM\\archive\\imbalanced_dataset")
    parser.add_argument("--validation-report", required=True,
                         help="Path to v3_volume_validation_report.csv")
    parser.add_argument("--manifest", required=False, default=None,
                         help="Path to dicom_3d_manifest.csv (optional, "
                              "not required by this script's own logic, "
                              "but accepted for completeness/future use).")
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_investigation",
                         help="Separate output directory for investigation "
                              "reports - never inside either dataset.")
    parser.add_argument("--sample-size", type=int, default=20,
                         help="Number of failures per category to "
                              "investigate. Ignored if --all is given.")
    parser.add_argument("--all", action="store_true",
                         help="Investigate every failure in both categories, "
                              "not just a sample.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.validation_report):
        print(f"ERROR: validation report not found: {args.validation_report}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("DICOM V3 FAILURE INVESTIGATION - READ-ONLY")
    print("=" * 70)

    # --- Load failures from the validation report ---
    spacing_failures = load_failures(args.validation_report, "inconsistent spacing")
    mono_failures = load_failures(args.validation_report, "non-monotonic z-ordering")
    print(f"\nSpacing failures found in validation report: {len(spacing_failures)}")
    print(f"Non-monotonic failures found in validation report: {len(mono_failures)}")

    selected_spacing = select_series(spacing_failures, args.sample_size, args.all)
    selected_mono = select_series(mono_failures, args.sample_size, args.all)
    print(f"\nSelected for investigation: {len(selected_spacing)} spacing, "
          f"{len(selected_mono)} non-monotonic "
          f"({'ALL' if args.all else f'sample, seed={RANDOM_SEED}'})")

    # --- Build the original-dataset index ONCE ---
    original_index = build_original_index(args.dataset_root)

    # --- Investigation 1-6: spacing failures ---
    print("\nInvestigating spacing failures against original dataset...")
    spacing_results = []
    excluded_in_gap_rows = []
    for row in selected_spacing:
        key = (str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))
        original_files = original_index.get(key, [])
        result = investigate_spacing_series(key, original_files)
        result["split"] = row.get("split")
        result["class"] = row.get("class")
        spacing_results.append(result)
        for gap in result.get("gaps", []):
            for ef in gap["excluded_files_in_gap"]:
                excluded_in_gap_rows.append({
                    "patient_id": key[0], "study_uid": key[1], "series_uid": key[2],
                    "gap_z_before": gap["z_before"], "gap_z_after": gap["z_after"],
                    "gap_size_mm": gap["gap_size_mm"],
                    "excluded_file_path": ef["path"], "excluded_z": ef["z"],
                    "modality": ef["modality"], "rescale_slope": ef["rescale_slope"],
                    "rescale_intercept": ef["rescale_intercept"],
                    "exclusion_reasons": ";".join(ef["exclusion_reasons"]),
                })

    total_gaps = sum(len(r.get("gaps", [])) for r in spacing_results)
    explained_gaps = sum(
        1 for r in spacing_results for g in r.get("gaps", [])
        if g["gap_explained_by_excluded_slice"]
    )
    print(f"  Total unusually-large gaps found: {total_gaps}")
    print(f"  Gaps explained by an excluded slice inside them: {explained_gaps} "
          f"({explained_gaps/total_gaps*100:.1f}%)" if total_gaps else "  No gaps found.")

    # --- Investigation 7-8: non-monotonic failures ---
    print("\nInvestigating non-monotonic failures against original dataset...")
    mono_results = []
    for row in selected_mono:
        key = (str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))
        original_files = original_index.get(key, [])
        result = investigate_nonmonotonic_series(key, original_files)
        result["split"] = row.get("split")
        result["class"] = row.get("class")
        mono_results.append(result)

    conclusion_counts = defaultdict(int)
    for r in mono_results:
        conclusion_counts[r.get("conclusion", "UNDETERMINED")] += 1
    print(f"  Conclusions: {dict(conclusion_counts)}")

    # --- Save outputs ---
    summary = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "validation_report": os.path.abspath(args.validation_report),
        "spacing_failures_total_in_report": len(spacing_failures),
        "spacing_failures_investigated": len(selected_spacing),
        "nonmonotonic_failures_total_in_report": len(mono_failures),
        "nonmonotonic_failures_investigated": len(selected_mono),
        "sample_mode": "ALL" if args.all else f"sample(size={args.sample_size}, seed={RANDOM_SEED})",
        "spacing_gaps_found": total_gaps,
        "spacing_gaps_explained_by_excluded_slice": explained_gaps,
        "nonmonotonic_conclusion_counts": dict(conclusion_counts),
    }
    with open(os.path.join(args.output, "investigation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    spacing_rows = []
    for r in spacing_results:
        spacing_rows.append({
            "patient_id": r["patient_id"], "study_uid": r["study_uid"], "series_uid": r["series_uid"],
            "split": r.get("split"), "class": r.get("class"),
            "original_file_count": r["original_file_count"],
            "included_file_count": r["included_file_count"],
            "excluded_file_count": r["excluded_file_count"],
            "num_gaps_found": len(r.get("gaps", [])),
            "num_gaps_explained": sum(1 for g in r.get("gaps", []) if g["gap_explained_by_excluded_slice"]),
        })
    pd.DataFrame(spacing_rows).to_csv(
        os.path.join(args.output, "spacing_failure_investigation.csv"), index=False
    )

    mono_rows = []
    for r in mono_results:
        mono_rows.append({
            "patient_id": r["patient_id"], "study_uid": r["study_uid"], "series_uid": r["series_uid"],
            "split": r.get("split"), "class": r.get("class"),
            "original_file_count": r["original_file_count"],
            "included_file_count": r["included_file_count"],
            "excluded_file_count": r["excluded_file_count"],
            "conclusion": r.get("conclusion"),
            "confidence": r.get("confidence", ""),
            "evidence": json.dumps(r.get("evidence", "")) if not isinstance(r.get("evidence"), str) else r.get("evidence", ""),
        })
    pd.DataFrame(mono_rows).to_csv(
        os.path.join(args.output, "nonmonotonic_failure_investigation.csv"), index=False
    )

    if excluded_in_gap_rows:
        pd.DataFrame(excluded_in_gap_rows).to_csv(
            os.path.join(args.output, "excluded_slices_in_gaps.csv"), index=False
        )

    print(f"\nReports saved to: {args.output}")
    print("investigation_summary.json, spacing_failure_investigation.csv, "
          "nonmonotonic_failure_investigation.csv" +
          (", excluded_slices_in_gaps.csv" if excluded_in_gap_rows else ""))
    print("\nNeither dataset was modified - this script only read files.")


if __name__ == "__main__":
    main()
