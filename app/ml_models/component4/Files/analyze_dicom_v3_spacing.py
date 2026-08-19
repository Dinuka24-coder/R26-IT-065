"""
analyze_dicom_v3_spacing.py

STANDALONE, READ-ONLY deep analysis of the 459 spacing-failure and 90
non-monotonic-failure series, against the real original DICOM dataset.

CRITICAL REFRAMING FROM THE PRIOR INVESTIGATION: the earlier
investigate_dicom_v3_failures.py run found that 100% of both failure
categories (459/459 spacing, 90/90 non-monotonic) have ZERO excluded
files - every file belonging to each series' (patient_id, study_uid,
series_uid) grouping is already fully included. This directly
falsifies the "V2 filtering excluded interior slices" hypothesis for
these series - there is nothing excluded to explain any gap. This
script therefore does NOT re-check the excluded-file-in-gap question
(already answered, negatively, at 100% coverage) - it instead tests
the remaining evidence source: whether gaps are consistent with
missing slices via the integer-multiple-of-dominant-spacing signature,
using richer acquisition metadata to help distinguish that from
genuinely coarse/intentional acquisition protocols.

READ-ONLY. Reads:
    - D:/DICOM/archive/imbalanced_dataset (DICOM metadata only,
      stop_before_pixels=True - no pixel decoding, no modification)
    - v3_volume_validation_report.csv (to get the 459/90 series identities
      and the 90 series' full failure_reason text, which contains the
      real z-position sequences)
Writes ONLY to a new, separate output directory - never into either
existing dataset directory, never into any existing report file.

Never calls shutil.copy/move, os.rename, os.remove, Path.unlink, or any
equivalent write/delete operation against a dataset path.

Usage:
    python analyze_dicom_v3_spacing.py \
        --dataset-root "D:/DICOM/archive/imbalanced_dataset" \
        --validation-report "D:/DICOM/archive/dicom_3d_dataset/v3_volume_validation_report.csv" \
        --output "D:/DICOM/archive/dicom_3d_spacing_analysis"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pydicom

# --- Documented, reasoned thresholds - stated explicitly, not buried ---

# A ratio (gap / dominant_spacing) is treated as "approximately an
# integer" if it's within this fraction of the nearest integer >= 2.
# 0.15 = 15% tolerance - loose enough for real floating-point noise in
# ImagePositionPatient, tight enough that a ratio of e.g. 2.4 or 1.6
# would NOT count as "approximately 2".
INTEGER_RATIO_TOLERANCE = 0.15

# Gap-flagging threshold, IDENTICAL to the prior investigation script,
# reused for consistency: an included-slice gap is "unusually large" if
# it exceeds 2x the series' own median gap.
GAP_RELATIVE_THRESHOLD = 2.0

# A dominant spacing at or below this value is treated as "a plausible
# fine-grained diagnostic CT increment" for classification purposes -
# a stated, reasoned cutoff (typical diagnostic chest CT is 1-5mm,
# occasionally up to ~10mm for older/thick-slice protocols), not an
# empirically-derived constant from this project's own data.
PLAUSIBLE_FINE_GRAINED_SPACING_MM = 10.0

# Optional acquisition-context tags - recorded as None/unavailable if
# absent, never causes a failure.
OPTIONAL_TAGS = [
    "SliceThickness", "SpacingBetweenSlices", "AcquisitionNumber",
    "AcquisitionDate", "AcquisitionTime", "SeriesNumber", "StudyID",
    "ProtocolName", "SeriesDescription", "ScanningSequence",
    "SequenceVariant", "Manufacturer", "ManufacturerModelName",
]


def safe_str(ds, tag):
    val = getattr(ds, tag, None)
    if val is None:
        return None
    return str(val)


def build_original_index(dataset_root: str) -> dict:
    """One pass over the original dataset, metadata only. Returns
    dict keyed by (patient_id, study_uid, series_uid) -> list of
    per-file metadata dicts, including the full optional acquisition
    tag set.
    """
    print(f"Building metadata index of {dataset_root} (one pass, metadata only)...")
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
                "instance_number": int(getattr(ds, "InstanceNumber")) if hasattr(ds, "InstanceNumber") else None,
                "image_position": [float(v) for v in ds.ImagePositionPatient] if hasattr(ds, "ImagePositionPatient") else None,
                "image_orientation": [float(v) for v in ds.ImageOrientationPatient] if hasattr(ds, "ImageOrientationPatient") else None,
                "pixel_spacing": [float(v) for v in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else None,
            }
            for tag in OPTIONAL_TAGS:
                meta[tag] = safe_str(ds, tag)

            key = (str(patient_id), str(study_uid), str(series_uid))
            index[key].append(meta)

    print(f"Indexed {total} files ({unreadable} unreadable). "
          f"{len(index)} distinct (patient, study, series) groups found.")
    return dict(index)


def compute_dominant_spacing(gaps: np.ndarray, round_to: float = 0.1) -> float:
    """Dominant spacing = the MODE of gap values, rounded to round_to
    mm for grouping (real ImagePositionPatient values carry floating-
    point noise; without rounding, a mode over raw floats would almost
    never repeat). Distinct from median, which the per-series report
    also includes separately.
    """
    if len(gaps) == 0:
        return float("nan")
    rounded = np.round(gaps / round_to) * round_to
    counts = Counter(rounded.tolist())
    return float(counts.most_common(1)[0][0])


def analyze_spacing_series(key: tuple, files: list[dict]) -> dict:
    with_pos = [f for f in files if f["image_position"] is not None]
    ordered = sorted(with_pos, key=lambda f: f["image_position"][2])
    n_slices = len(ordered)

    result = {
        "patient_id": key[0], "study_uid": key[1], "series_uid": key[2],
        "num_slices": n_slices,
    }

    # Acquisition-context metadata - from the first file (assumed
    # consistent across the series; not independently re-verified here).
    if ordered:
        first = ordered[0]
        result["slice_thickness"] = first.get("SliceThickness")
        result["pixel_spacing"] = first["pixel_spacing"]
        result["image_orientation"] = first["image_orientation"]
        for tag in OPTIONAL_TAGS:
            result[tag.lower()] = first.get(tag)

    if n_slices < 3:
        result["classification"] = "INSUFFICIENT_DATA"
        result["classification_evidence"] = f"Only {n_slices} slice(s) with position - cannot compute meaningful spacing statistics."
        return result

    z = np.array([f["image_position"][2] for f in ordered])
    gaps = np.abs(np.diff(z))

    median_spacing = float(np.median(gaps))
    mean_spacing = float(np.mean(gaps))
    dominant_spacing = compute_dominant_spacing(gaps)
    min_spacing = float(gaps.min())
    max_gap = float(gaps.max())

    large_gap_threshold = median_spacing * GAP_RELATIVE_THRESHOLD
    large_gaps = gaps[gaps > large_gap_threshold] if median_spacing > 0 else gaps[gaps > 0]

    # Integer-ratio test against dominant spacing
    integer_multiple_gaps = 0
    ratios = []
    if dominant_spacing > 0:
        for g in large_gaps:
            ratio = g / dominant_spacing
            nearest_int = round(ratio)
            ratios.append(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int <= INTEGER_RATIO_TOLERANCE:
                integer_multiple_gaps += 1

    result.update({
        "median_spacing_mm": median_spacing,
        "mean_spacing_mm": mean_spacing,
        "dominant_spacing_mm": dominant_spacing,
        "min_spacing_mm": min_spacing,
        "max_gap_mm": max_gap,
        "num_gaps": int(len(gaps)),
        "num_unusually_large_gaps": int(len(large_gaps)),
        "num_gaps_integer_multiple_of_dominant": int(integer_multiple_gaps),
        "pct_large_gaps_integer_multiple": (
            integer_multiple_gaps / len(large_gaps) * 100 if len(large_gaps) else 0.0
        ),
    })

    # --- Classification, evidence-based, documented reasoning ---
    if len(large_gaps) == 0:
        result["classification"] = "CONSISTENT_SPACING_BUT_OTHER_GEOMETRY_ISSUE"
        result["classification_evidence"] = (
            "No unusually large gaps found relative to this series' own "
            "median spacing - the reported build_volume() spacing failure "
            "is not explained by a small number of large gaps; likely "
            "broadly elevated variance across many small differences instead."
        )
    else:
        pct_integer = integer_multiple_gaps / len(large_gaps)
        fine_grained = dominant_spacing <= PLAUSIBLE_FINE_GRAINED_SPACING_MM
        if pct_integer >= 0.5 and fine_grained:
            result["classification"] = "LIKELY_MISSING_SLICES"
            result["classification_evidence"] = (
                f"{integer_multiple_gaps}/{len(large_gaps)} "
                f"({pct_integer*100:.0f}%) large gaps are approximately "
                f"integer multiples of the dominant spacing "
                f"({dominant_spacing}mm, a plausible fine-grained "
                f"diagnostic increment) - consistent with slices missing "
                f"from an otherwise regular acquisition."
            )
        elif 0.2 <= pct_integer < 0.5 and fine_grained:
            result["classification"] = "POSSIBLE_MISSING_SLICES"
            result["classification_evidence"] = (
                f"{integer_multiple_gaps}/{len(large_gaps)} "
                f"({pct_integer*100:.0f}%) large gaps are approximately "
                f"integer multiples of dominant spacing - a real but "
                f"partial signal, not conclusive."
            )
        elif pct_integer < 0.2:
            result["classification"] = "LIKELY_INTENTIONAL_VARIABLE_SPACING"
            result["classification_evidence"] = (
                f"Only {integer_multiple_gaps}/{len(large_gaps)} "
                f"({pct_integer*100:.0f}%) large gaps align with integer "
                f"multiples of dominant spacing - gaps do not cluster "
                f"around a single underlying increment, more consistent "
                f"with genuinely variable real acquisition spacing than "
                f"missing data."
            )
        else:
            result["classification"] = "UNKNOWN"
            result["classification_evidence"] = (
                f"Evidence mixed: {pct_integer*100:.0f}% integer-multiple "
                f"gaps, dominant_spacing={dominant_spacing}mm "
                f"(fine_grained={fine_grained}) - does not cleanly fit "
                f"the other categories."
            )

    return result


def parse_z_from_validation_reason(reason_text: str) -> list[float] | None:
    match = re.search(r'z-positions: (\[.*\])', reason_text)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except Exception:
            return None
    return None


def analyze_undetermined_series(key: tuple, files: list[dict], z_from_report: list[float] | None) -> dict:
    with_both = [f for f in files if f["image_position"] is not None and f["instance_number"] is not None]
    ordered = sorted(with_both, key=lambda f: f["instance_number"])

    result = {
        "patient_id": key[0], "study_uid": key[1], "series_uid": key[2],
        "num_slices": len(ordered),
    }

    if len(ordered) < 2:
        result["diagnosis"] = "GENUINELY_AMBIGUOUS"
        result["evidence"] = "Fewer than 2 slices with both InstanceNumber and position."
        return result

    z = [f["image_position"][2] for f in ordered]
    diffs = np.diff(z)
    exact_ties = int((diffs == 0).sum())

    acquisition_numbers = set(f.get("AcquisitionNumber") for f in ordered if f.get("AcquisitionNumber"))
    instance_numbers = [f["instance_number"] for f in ordered]
    duplicate_instance_numbers = len(instance_numbers) - len(set(instance_numbers))

    result["exact_z_ties"] = exact_ties
    result["distinct_acquisition_numbers"] = len(acquisition_numbers)
    result["acquisition_numbers_seen"] = sorted(acquisition_numbers) if acquisition_numbers else None
    result["duplicate_instance_numbers"] = duplicate_instance_numbers

    if exact_ties > 0 and len(ordered) <= 4:
        result["diagnosis"] = "DUPLICATE_SLICES"
        result["evidence"] = (
            f"{exact_ties} exact-tie z-position pair(s) found in a small "
            f"({len(ordered)}-slice) series - consistent with a literal "
            f"duplicate slice export rather than genuine multi-acquisition."
        )
    elif len(acquisition_numbers) > 1:
        result["diagnosis"] = "REPEATED_ACQUISITION"
        result["evidence"] = (
            f"{len(acquisition_numbers)} distinct AcquisitionNumber values "
            f"found within one series - direct metadata evidence of "
            f"multiple separate acquisitions grouped under one "
            f"SeriesInstanceUID, independent of the z-position pattern alone."
        )
    elif duplicate_instance_numbers > 0:
        result["diagnosis"] = "INCORRECT_INSTANCE_ORDERING"
        result["evidence"] = (
            f"{duplicate_instance_numbers} duplicate InstanceNumber "
            f"value(s) found - InstanceNumber does not uniquely identify "
            f"slice order in this series."
        )
    else:
        result["diagnosis"] = "GENUINELY_AMBIGUOUS"
        result["evidence"] = (
            "No exact ties, no multiple AcquisitionNumbers, no duplicate "
            "InstanceNumbers found - the non-monotonicity is not "
            "explained by any of the specific mechanisms checked; real "
            "cause undetermined from available metadata."
        )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--output", default=r"D:/DICOM/archive/dicom_3d_spacing_analysis")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.validation_report):
        print(f"ERROR: validation report not found: {args.validation_report}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("DICOM V3 SPACING/NON-MONOTONIC DEEP ANALYSIS - READ-ONLY")
    print("=" * 70)

    df = pd.read_csv(args.validation_report)
    spacing_rows = df[df["failure_category"] == "inconsistent spacing"].to_dict("records")
    mono_rows = df[df["failure_category"] == "non-monotonic z-ordering"].to_dict("records")
    print(f"Spacing-failure series in report: {len(spacing_rows)}")
    print(f"Non-monotonic-failure series in report: {len(mono_rows)}")

    original_index = build_original_index(args.dataset_root)

    # --- Spacing analysis ---
    print("\nAnalyzing spacing-failure series...")
    spacing_results = []
    gap_rows = []
    for row in spacing_rows:
        key = (str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))
        files = original_index.get(key, [])
        result = analyze_spacing_series(key, files)
        result["split"] = row.get("split")
        result["class"] = row.get("class")
        spacing_results.append(result)

        with_pos = [f for f in files if f["image_position"] is not None]
        ordered = sorted(with_pos, key=lambda f: f["image_position"][2])
        if len(ordered) >= 3:
            z = np.array([f["image_position"][2] for f in ordered])
            gaps = np.abs(np.diff(z))
            median_spacing = float(np.median(gaps))
            dominant = result.get("dominant_spacing_mm", float("nan"))
            threshold = median_spacing * GAP_RELATIVE_THRESHOLD
            for i, g in enumerate(gaps):
                if g > threshold and median_spacing > 0:
                    ratio = g / dominant if dominant else float("nan")
                    nearest_int = round(ratio) if not np.isnan(ratio) else None
                    is_integer_multiple = (
                        nearest_int is not None and nearest_int >= 2 and
                        abs(ratio - nearest_int) / nearest_int <= INTEGER_RATIO_TOLERANCE
                    )
                    gap_rows.append({
                        "patient_id": key[0], "study_uid": key[1], "series_uid": key[2],
                        "z_before": z[i], "z_after": z[i + 1], "gap_mm": g,
                        "series_median_spacing_mm": median_spacing,
                        "series_dominant_spacing_mm": dominant,
                        "ratio_gap_to_dominant": ratio,
                        "is_approx_integer_multiple": is_integer_multiple,
                    })

    total_large_gaps = len(gap_rows)
    integer_gaps = sum(1 for g in gap_rows if g["is_approx_integer_multiple"])
    print(f"  Total unusually-large gaps analyzed: {total_large_gaps}")
    print(f"  Gaps approx. integer multiple of dominant spacing: {integer_gaps} "
          f"({integer_gaps/total_large_gaps*100:.1f}%)" if total_large_gaps else "  N/A")

    classification_counts = Counter(r.get("classification", "UNKNOWN") for r in spacing_results)

    dominant_spacing_values = [
        r["dominant_spacing_mm"] for r in spacing_results
        if "dominant_spacing_mm" in r and not np.isnan(r["dominant_spacing_mm"])
    ]
    dominant_spacing_hist = Counter(round(v, 2) for v in dominant_spacing_values)

    # --- Non-monotonic UNDETERMINED deep-dive ---
    print("\nAnalyzing the UNDETERMINED non-monotonic series...")
    undetermined_rows = [r for r in mono_rows if "UNDETERMINED" in str(r.get("failure_reason", "")) or True]
    # Filter using the ORIGINAL investigation's conclusion is not in this
    # validation report - re-derive UNDETERMINED status the same way the
    # prior script did, using the z-position text embedded in
    # failure_reason, since validation_report itself doesn't carry
    # the earlier investigation's conclusion label.
    undetermined_results = []
    for row in mono_rows:
        key = (str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))
        files = original_index.get(key, [])
        z_from_report = parse_z_from_validation_reason(str(row.get("failure_reason", "")))
        # Re-derive run structure to identify UNDETERMINED cases consistent
        # with the prior investigation's own logic (>=2 long runs = likely
        # multiple acquisitions; otherwise undetermined-candidate).
        with_both = [f for f in files if f["image_position"] is not None and f["instance_number"] is not None]
        ordered = sorted(with_both, key=lambda f: f["instance_number"])
        if len(ordered) < 3:
            continue
        z = [f["image_position"][2] for f in ordered]
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
        long_runs = [r for r in runs if r >= 3]
        if len(long_runs) < 2:
            result = analyze_undetermined_series(key, files, z_from_report)
            result["split"] = row.get("split")
            result["class"] = row.get("class")
            undetermined_results.append(result)

    print(f"  UNDETERMINED series re-identified and analyzed: {len(undetermined_results)}")
    diagnosis_counts = Counter(r.get("diagnosis", "GENUINELY_AMBIGUOUS") for r in undetermined_results)
    resolved = sum(c for d, c in diagnosis_counts.items() if d != "GENUINELY_AMBIGUOUS")
    still_ambiguous = diagnosis_counts.get("GENUINELY_AMBIGUOUS", 0)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total spacing-failure series: {len(spacing_rows)}")
    print(f"Total large gaps: {total_large_gaps}")
    print(f"Integer-multiple gaps: {integer_gaps}")
    print(f"Percentage: {integer_gaps/total_large_gaps*100:.1f}%" if total_large_gaps else "N/A")
    print(f"\nDominant spacing distribution (rounded to 0.01mm):")
    for val, count in dominant_spacing_hist.most_common(15):
        print(f"  {val}mm: {count}")
    print(f"\nCategories:")
    for cat, count in classification_counts.most_common():
        print(f"  {cat}: {count} ({count/len(spacing_results)*100:.1f}%)")
    print(f"\nUNDETERMINED non-monotonic: {len(undetermined_results)}")
    print(f"Resolved: {resolved}")
    print(f"Still ambiguous: {still_ambiguous}")
    for d, c in diagnosis_counts.items():
        print(f"  {d}: {c}")

    # --- Save outputs ---
    summary = {
        "total_spacing_failure_series": len(spacing_rows),
        "total_large_gaps": total_large_gaps,
        "integer_multiple_gaps": integer_gaps,
        "integer_multiple_pct": integer_gaps / total_large_gaps * 100 if total_large_gaps else None,
        "dominant_spacing_distribution": {str(k): v for k, v in dominant_spacing_hist.most_common()},
        "classification_counts": dict(classification_counts),
        "undetermined_total": len(undetermined_results),
        "undetermined_resolved": resolved,
        "undetermined_still_ambiguous": still_ambiguous,
        "undetermined_diagnosis_counts": dict(diagnosis_counts),
        "integer_ratio_tolerance": INTEGER_RATIO_TOLERANCE,
        "gap_relative_threshold": GAP_RELATIVE_THRESHOLD,
        "plausible_fine_grained_spacing_mm": PLAUSIBLE_FINE_GRAINED_SPACING_MM,
    }
    with open(os.path.join(args.output, "spacing_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    pd.DataFrame(spacing_results).to_csv(
        os.path.join(args.output, "spacing_series_analysis.csv"), index=False
    )
    pd.DataFrame(gap_rows).to_csv(
        os.path.join(args.output, "spacing_gap_analysis.csv"), index=False
    )
    pd.DataFrame(undetermined_results).to_csv(
        os.path.join(args.output, "undetermined_nonmonotonic_analysis.csv"), index=False
    )

    print(f"\nReports saved to: {args.output}")
    print("Neither dataset was modified - this script only read files.")


if __name__ == "__main__":
    main()
