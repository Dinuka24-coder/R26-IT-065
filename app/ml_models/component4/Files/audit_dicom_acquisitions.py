"""
audit_dicom_acquisitions.py

STANDALONE, READ-ONLY, dataset-wide audit: does one SeriesInstanceUID
ever contain multiple genuinely separate CT acquisitions, and would an
Acquisition-level grouping layer below SeriesInstanceUID be justified?

Motivated by a real confirmed case (Lung_Dx-E0003, series_id
52e51ee5-fe51-4e9d-9aa1-05d3e4eaf71a) where 106 files sharing one
SeriesInstanceUID/StudyInstanceUID/FrameOfReferenceUID split cleanly by
AcquisitionNumber into 3 independently-monotonic acquisitions (36/39/31
slices) with corroborating distinct AcquisitionTime values and
non-overlapping InstanceNumber ranges. This audit determines whether
that is an isolated case or a broader dataset pattern - it does NOT
assume the answer.

Imports and calls the REAL build_volume() UNMODIFIED to evaluate each
candidate acquisition-level group - never reimplements or approximates
its geometry logic, per instruction.

READ-ONLY. Never writes, renames, moves, or deletes any DICOM file.
Never modifies series.py, volume.py, the API, the frontend, or any
existing manifest. Only writes new files into --output.

Every reported number distinguishes:
    DIRECT EVIDENCE   - read straight from a DICOM tag
    COMPUTED RESULT    - calculated by this script from direct evidence
    INFERENCE          - a conclusion/judgment based on the evidence,
                          not itself a measured fact

Usage:
    python audit_dicom_acquisitions.py \
        --dataset-root "D:/DICOM/archive/dicom_3d_working_dataset" \
        --project-root "D:/R26-IT-065" \
        --output "D:/DICOM/archive/dicom_acquisition_audit"

    --dataset-root should point at the flat, full working copy (18,500
    files) rather than the organized dicom_3d_final tree, so that EVERY
    file - usable, needs_review, and unusable alike - is included in
    the audit; the multi-acquisition question is about raw DICOM
    grouping, not about the existing usability classification.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict, Counter

import numpy as np
import pydicom

NAMED_CASES_OF_INTEREST = {"Lung_Dx-E0003", "Lung_Dx-B0025", "Lung_Dx-A0198"}


def safe_str(ds, tag, default=None):
    v = getattr(ds, tag, default)
    return str(v) if v is not None else default


def safe_float_list(ds, tag):
    v = getattr(ds, tag, None)
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def read_series_relevant_fields(path):
    """Reads exactly the fields requested, safely - a missing optional
    tag returns None for that field, never aborts the whole file.
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "path": path,
        "PatientID": safe_str(ds, "PatientID"),
        "StudyInstanceUID": safe_str(ds, "StudyInstanceUID"),
        "SeriesInstanceUID": safe_str(ds, "SeriesInstanceUID"),
        "SeriesNumber": safe_str(ds, "SeriesNumber"),
        "SeriesDescription": safe_str(ds, "SeriesDescription"),
        "ProtocolName": safe_str(ds, "ProtocolName"),
        "Modality": safe_str(ds, "Modality"),
        "FrameOfReferenceUID": safe_str(ds, "FrameOfReferenceUID"),
        "AcquisitionNumber": safe_str(ds, "AcquisitionNumber"),
        "AcquisitionTime": safe_str(ds, "AcquisitionTime"),
        "InstanceNumber": getattr(ds, "InstanceNumber", None),
        "ImagePositionPatient": safe_float_list(ds, "ImagePositionPatient"),
        "ImageOrientationPatient": safe_float_list(ds, "ImageOrientationPatient"),
        "SliceThickness": safe_str(ds, "SliceThickness"),
        "SpacingBetweenSlices": safe_str(ds, "SpacingBetweenSlices"),
        "PixelSpacing": safe_float_list(ds, "PixelSpacing"),
        "Rows": getattr(ds, "Rows", None),
        "Columns": getattr(ds, "Columns", None),
        "ImageType": list(getattr(ds, "ImageType", []) or []),
        "Manufacturer": safe_str(ds, "Manufacturer"),
        "ManufacturerModelName": safe_str(ds, "ManufacturerModelName"),
        "ConvolutionKernel": safe_str(ds, "ConvolutionKernel"),
        "KVP": safe_str(ds, "KVP"),
    }


def evaluate_group_with_real_build_volume(datasets, build_volume, VolumeGeometryError):
    """Calls the REAL, unmodified build_volume() - this is COMPUTED
    RESULT, not a re-implementation of its logic.
    """
    try:
        vol = build_volume(datasets)
        return True, "", vol
    except VolumeGeometryError as exc:
        return False, str(exc), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--project-root", required=True,
                         help="Project root containing app/ml_models/component4/dicom/volume.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: --dataset-root not found: {args.dataset_root}", file=sys.stderr)
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
    print("DICOM ACQUISITION-LEVEL GROUPING AUDIT - READ-ONLY")
    print("=" * 70)

    # --- Step 1: recursive scan ----------------------------------------
    print(f"\nScanning {args.dataset_root} recursively for .dcm files...")
    all_files = []
    for root, _dirs, files in os.walk(args.dataset_root):
        for fname in files:
            if fname.lower().endswith(".dcm"):
                all_files.append(os.path.join(root, fname))
    print(f"Total .dcm files found: {len(all_files)}")

    total_files_scanned = 0
    total_ct_files = 0
    read_errors = []
    series_groups = defaultdict(list)  # SeriesInstanceUID -> [record, ...]

    for i, path in enumerate(all_files):
        if (i + 1) % 2000 == 0:
            print(f"  ...scanned {i + 1}/{len(all_files)}")
        try:
            rec = read_series_relevant_fields(path)
        except Exception as exc:
            read_errors.append({"path": path, "error": str(exc)})
            continue
        total_files_scanned += 1
        if rec["Modality"] == "CT":
            total_ct_files += 1
            series_groups[rec["SeriesInstanceUID"]].append(rec)

    print(f"\nFiles successfully read: {total_files_scanned}")
    print(f"Read errors (skipped, not fatal): {len(read_errors)}")
    print(f"CT files: {total_ct_files}")
    print(f"Distinct SeriesInstanceUIDs (CT only): {len(series_groups)}")

    # --- Step 2/3: classify each series by AcquisitionNumber behavior --
    series_summary = []
    single_acq = 0
    multi_acq = 0
    missing_acq = 0
    multi_acq_series_keys = []

    for series_uid, recs in series_groups.items():
        acq_values = [r["AcquisitionNumber"] for r in recs]
        missing_count = sum(1 for v in acq_values if v is None)
        present_values = [v for v in acq_values if v is not None]
        unique_values = sorted(set(present_values), key=lambda x: (len(x), x))

        if missing_count == len(recs):
            classification = "ACQUISITION_NUMBER_UNAVAILABLE"
            missing_acq += 1
        elif missing_count > 0:
            classification = "PARTIAL_MISSING_ACQUISITION_NUMBER"
        elif len(unique_values) == 1:
            classification = "SINGLE_ACQUISITION"
            single_acq += 1
        else:
            classification = "MULTIPLE_ACQUISITIONS"
            multi_acq += 1
            multi_acq_series_keys.append(series_uid)

        series_summary.append({
            "patient_id": recs[0]["PatientID"],
            "study_instance_uid": recs[0]["StudyInstanceUID"],
            "series_instance_uid": series_uid,
            "series_number": recs[0]["SeriesNumber"],
            "series_description": recs[0]["SeriesDescription"],
            "protocol_name": recs[0]["ProtocolName"],
            "num_files": len(recs),
            "num_unique_acquisition_numbers": len(unique_values),
            "acquisition_number_values": ";".join(unique_values),
            "num_missing_acquisition_number": missing_count,
            "classification": classification,
        })

    print(f"\n=== DATASET-WIDE ACQUISITION CLASSIFICATION (COMPUTED RESULT) ===")
    print(f"Total DICOM files (all, including non-CT): {len(all_files)}")
    print(f"Total CT files: {total_ct_files}")
    print(f"Total SeriesInstanceUIDs (CT): {len(series_groups)}")
    print(f"Series with exactly 1 AcquisitionNumber: {single_acq}")
    print(f"Series with >1 AcquisitionNumber: {multi_acq}")
    print(f"Series with missing AcquisitionNumber (all files): {missing_acq}")
    print(f"Series with PARTIALLY missing AcquisitionNumber: "
          f"{sum(1 for s in series_summary if s['classification']=='PARTIAL_MISSING_ACQUISITION_NUMBER')}")
    if len(series_groups):
        print(f"Percentage multi-acquisition: {multi_acq / len(series_groups) * 100:.2f}%")

    # --- Step 4/5: for every multi-acquisition series, evaluate each
    # acquisition group with the REAL build_volume(), and detect edge
    # cases from section 5 of the request ------------------------------
    acquisition_group_rows = []
    edge_cases = []

    field_corroboration = Counter()  # for question 6 statistics

    for series_uid in multi_acq_series_keys:
        recs = series_groups[series_uid]
        by_acq = defaultdict(list)
        for r in recs:
            by_acq[r["AcquisitionNumber"]].append(r)

        # Series-level corroboration check (question 6): does each
        # acquisition group have its OWN distinct AcquisitionTime, with
        # no two groups sharing a time? Computed once per series here,
        # not per group - comparing group-count to group-count correctly.
        times_per_group = []
        for acq_num, group in by_acq.items():
            group_times = set(r["AcquisitionTime"] for r in group if r["AcquisitionTime"])
            times_per_group.append(group_times)
        all_times_seen = set()
        for t in times_per_group:
            all_times_seen |= t
        distinct_time_per_group = (
            len(all_times_seen) == len(by_acq)
            and all(len(t) >= 1 for t in times_per_group)
        )
        if distinct_time_per_group:
            field_corroboration["distinct_acquisition_time_per_group"] += 1

        # Series-level: is orientation/FrameOfReferenceUID the SAME
        # across the different acquisition groups (not just within one
        # group, which is nearly tautological)?
        series_orientations = set(
            tuple(r["ImageOrientationPatient"]) for r in recs
            if r["ImageOrientationPatient"] is not None
        )
        series_frame_refs = set(r["FrameOfReferenceUID"] for r in recs)
        if len(series_orientations) <= 1:
            field_corroboration["consistent_orientation_across_groups"] += 1
        if len(series_frame_refs) <= 1:
            field_corroboration["consistent_frame_of_reference_across_groups"] += 1

        # Edge case: same AcquisitionNumber used for what instance-number
        # evidence suggests are actually separate sweeps (large InstanceNumber
        # gap within one nominal acquisition group) - detected, not assumed.
        for acq_num, group in by_acq.items():
            group_sorted = sorted(group, key=lambda r: r["InstanceNumber"] if r["InstanceNumber"] is not None else 0)
            inst_numbers = [r["InstanceNumber"] for r in group_sorted if r["InstanceNumber"] is not None]
            if len(inst_numbers) >= 2:
                gaps = np.diff(sorted(inst_numbers))
                if len(gaps) and gaps.max() > 3 * np.median(gaps) and gaps.max() > 10:
                    edge_cases.append({
                        "series_instance_uid": series_uid, "acquisition_number": acq_num,
                        "edge_case": "LARGE_INSTANCE_NUMBER_GAP_WITHIN_ONE_ACQUISITION",
                        "detail": f"InstanceNumber gaps within this AcquisitionNumber: max gap "
                                  f"{gaps.max()}, median {np.median(gaps):.1f} - possible undetected "
                                  f"sub-sweep sharing one AcquisitionNumber value."
                    })

        # Edge case: overlapping InstanceNumber ranges between acquisition groups
        ranges = {}
        for acq_num, group in by_acq.items():
            inst = [r["InstanceNumber"] for r in group if r["InstanceNumber"] is not None]
            if inst:
                ranges[acq_num] = (min(inst), max(inst))
        acq_nums_list = list(ranges.keys())
        for i in range(len(acq_nums_list)):
            for j in range(i + 1, len(acq_nums_list)):
                a, b = ranges[acq_nums_list[i]], ranges[acq_nums_list[j]]
                if max(a[0], b[0]) <= min(a[1], b[1]):
                    edge_cases.append({
                        "series_instance_uid": series_uid,
                        "acquisition_number": f"{acq_nums_list[i]} vs {acq_nums_list[j]}",
                        "edge_case": "OVERLAPPING_INSTANCE_NUMBER_RANGES_BETWEEN_ACQUISITIONS",
                        "detail": f"ranges {a} and {b} overlap - AcquisitionNumber split may not "
                                  f"correspond to a clean InstanceNumber separation here."
                    })

        # Edge case: differing FrameOfReferenceUID across acquisitions within one series
        fors = set(r["FrameOfReferenceUID"] for r in recs)
        if len(fors) > 1:
            edge_cases.append({
                "series_instance_uid": series_uid, "acquisition_number": "ALL",
                "edge_case": "MULTIPLE_FRAME_OF_REFERENCE_UIDS_WITHIN_ONE_SERIES",
                "detail": f"FrameOfReferenceUID values: {fors} - acquisitions may not share the "
                          f"same physical coordinate system, unlike the confirmed E0003 case."
            })

        # Now evaluate each acquisition group with the REAL build_volume().
        # IMPORTANT: the datasets read during the initial scan used
        # stop_before_pixels=True (correct/efficient for metadata grouping)
        # - build_volume() legitimately needs real pixel data for HU
        # conversion, so each candidate group's files are RE-READ here,
        # in full, only for the groups actually being evaluated - not
        # for the whole dataset, keeping this efficient while still
        # giving build_volume() what it genuinely requires.
        for acq_num, group in by_acq.items():
            group_sorted_by_inst = sorted(
                [r for r in group if r["InstanceNumber"] is not None],
                key=lambda r: r["InstanceNumber"]
            )
            datasets = [pydicom.dcmread(r["path"]) for r in group_sorted_by_inst]

            z_positions = [r["ImagePositionPatient"][2] for r in group_sorted_by_inst
                            if r["ImagePositionPatient"] is not None]
            z_monotonic = None
            if len(z_positions) >= 2:
                diffs = np.diff(z_positions)
                z_monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))

            acq_times = sorted(set(r["AcquisitionTime"] for r in group if r["AcquisitionTime"]))
            orientations = set(tuple(r["ImageOrientationPatient"]) for r in group
                                if r["ImageOrientationPatient"] is not None)
            pixel_spacings = set(tuple(r["PixelSpacing"]) for r in group if r["PixelSpacing"] is not None)
            rows_cols = set((r["Rows"], r["Columns"]) for r in group)
            slice_thicknesses = set(r["SliceThickness"] for r in group if r["SliceThickness"])
            spacing_between = set(r["SpacingBetweenSlices"] for r in group if r["SpacingBetweenSlices"])
            frame_refs = set(r["FrameOfReferenceUID"] for r in group)

            valid, reason, vol = evaluate_group_with_real_build_volume(
                datasets, build_volume, VolumeGeometryError
            )

            acquisition_group_rows.append({
                "patient_id": recs[0]["PatientID"],
                "study_instance_uid": recs[0]["StudyInstanceUID"],
                "series_instance_uid": series_uid,
                "acquisition_number": acq_num,
                "acquisition_time": ";".join(acq_times),
                "slice_count": len(group),
                "instance_min": min((r["InstanceNumber"] for r in group if r["InstanceNumber"] is not None), default=None),
                "instance_max": max((r["InstanceNumber"] for r in group if r["InstanceNumber"] is not None), default=None),
                "z_min": min(z_positions) if z_positions else None,
                "z_max": max(z_positions) if z_positions else None,
                "z_monotonic": z_monotonic,
                "orientation_consistent": len(orientations) <= 1,
                "pixel_spacing_consistent": len(pixel_spacings) <= 1,
                "rows_columns_consistent": len(rows_cols) <= 1,
                "slice_thickness_consistent": len(slice_thicknesses) <= 1,
                "spacing_between_slices_consistent": len(spacing_between) <= 1,
                "frame_of_reference_consistent": len(frame_refs) <= 1,
                "candidate_valid_volume": valid,
                "rejection_reason": reason,
            })


    # Non-overlapping InstanceNumber ranges - dataset-wide tally (question 6)
    non_overlapping_count = 0
    for series_uid in multi_acq_series_keys:
        recs = series_groups[series_uid]
        by_acq = defaultdict(list)
        for r in recs:
            by_acq[r["AcquisitionNumber"]].append(r)
        ranges = []
        for acq_num, group in by_acq.items():
            inst = [r["InstanceNumber"] for r in group if r["InstanceNumber"] is not None]
            if inst:
                ranges.append((min(inst), max(inst)))
        overlaps_found = False
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                if max(ranges[i][0], ranges[j][0]) <= min(ranges[i][1], ranges[j][1]):
                    overlaps_found = True
        if not overlaps_found and ranges:
            non_overlapping_count += 1

    print(f"\n=== MULTI-ACQUISITION SERIES: FIELD CORROBORATION (COMPUTED RESULT) ===")
    print(f"Multi-acquisition series found: {len(multi_acq_series_keys)}")
    if multi_acq_series_keys:
        print(f"  ...with distinct AcquisitionTime per group: "
              f"{field_corroboration['distinct_acquisition_time_per_group']}")
        print(f"  ...with non-overlapping InstanceNumber ranges: {non_overlapping_count}")
        print(f"  ...with consistent orientation across groups: "
              f"{field_corroboration['consistent_orientation_across_groups']}")
        print(f"  ...with consistent FrameOfReferenceUID across groups: "
              f"{field_corroboration['consistent_frame_of_reference_across_groups']}")

    valid_groups = sum(1 for r in acquisition_group_rows if r["candidate_valid_volume"])
    invalid_groups = len(acquisition_group_rows) - valid_groups
    print(f"\n=== ACQUISITION-LEVEL VOLUME VALIDITY (COMPUTED RESULT, via REAL build_volume()) ===")
    print(f"Total acquisition groups evaluated: {len(acquisition_group_rows)}")
    print(f"Valid (would form a real DicomVolume): {valid_groups}")
    print(f"Invalid (rejected by the same geometry rules as today): {invalid_groups}")

    print(f"\n=== EDGE CASES DETECTED (section 5 of request) ===")
    edge_case_counts = Counter(e["edge_case"] for e in edge_cases)
    for k, v in edge_case_counts.items():
        print(f"  {k}: {v}")

    # --- Named cases of interest ----------------------------------------
    print(f"\n=== NAMED CASES OF INTEREST ===")
    found_patients = set(s["patient_id"] for s in series_summary)
    for name in NAMED_CASES_OF_INTEREST:
        if name in found_patients:
            matches = [s for s in series_summary if s["patient_id"] == name]
            print(f"  {name}: FOUND, {len(matches)} series")
            for m in matches:
                print(f"    series {m['series_instance_uid'][-12:]}: {m['classification']}, "
                      f"{m['num_files']} files, AcquisitionNumbers={m['acquisition_number_values']}")
        else:
            print(f"  {name}: NOT FOUND in this dataset-root - cannot verify, not assumed.")

    # --- Write outputs ---------------------------------------------------
    with open(os.path.join(args.output, "series_summary.csv"), "w", newline="") as f:
        fieldnames = list(series_summary[0].keys()) if series_summary else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(series_summary)

    with open(os.path.join(args.output, "acquisition_groups.csv"), "w", newline="") as f:
        fieldnames = list(acquisition_group_rows[0].keys()) if acquisition_group_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(acquisition_group_rows)

    with open(os.path.join(args.output, "edge_cases.csv"), "w", newline="") as f:
        fieldnames = list(edge_cases[0].keys()) if edge_cases else \
            ["series_instance_uid", "acquisition_number", "edge_case", "detail"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edge_cases)

    with open(os.path.join(args.output, "read_errors.csv"), "w", newline="") as f:
        fieldnames = ["path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(read_errors)

    print(f"\nOutputs written to: {args.output}")
    print("  series_summary.csv, acquisition_groups.csv, edge_cases.csv, read_errors.csv")
    print("\nNo DICOM files, manifests, or source files were modified - read-only audit.")


if __name__ == "__main__":
    main()
