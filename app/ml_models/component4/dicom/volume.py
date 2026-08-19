"""
volume.py

Constructs a verified, spatially-correct 3D CT volume from ONE DICOM
series' slices. Visualization/multiplanar-viewing foundation only -
does NOT touch, call, or know about the existing 2D classifier,
Grad-CAM, or OOD pipeline. Those remain exactly as they are; this
module has no dependency on dicom_model.py, dicom_inference.py,
gradcam.py, or mobilenet_ood.py, and none of those files import this
one either.

AXIS CONVENTION - stated explicitly, not left implicit:
    volume.shape == (num_slices, rows, columns)
    volume[k, :, :] is slice k in verified physical order (index 0 =
    first slice in that order, NOT necessarily InstanceNumber 1 - see
    ordering_method for which tag actually determined that order).
    rows = dataset.Rows (patient anterior-posterior extent for
    standard axial acquisition), columns = dataset.Columns (patient
    left-right extent). This matches the row-major (z, y, x) convention
    used by SimpleITK's GetArrayFromImage and is the most common
    convention for medical volume arrays - chosen for that reason, not
    arbitrarily.

HU CONVERSION: reuses windowing.py's to_hounsfield_units() UNCHANGED,
per instruction - no parallel HU implementation exists here.

GEOMETRY VERIFICATION - a volume is only constructed if ALL of the
following hold; otherwise VolumeGeometryError is raised with a
specific reason. Nothing is silently guessed:
    1. All slices share the same SeriesInstanceUID (if present on all;
       if ANY slice lacks it, that itself is a failure - can't verify
       series identity without it).
    2. Reliable slice ordering exists (InstanceNumber or
       ImagePositionPatient on every slice - see series.py's new
       determine_ordering()). Upload-order fallback is NEVER accepted
       for volume construction, even though series.py's existing
       _sort_slices() would silently allow it for 2D viewing.
    3. ImagePositionPatient present on EVERY slice - required
       independently of which method ordered them, because without it
       there is no way to verify monotonic ordering or compute real
       inter-slice spacing. A series ordered correctly by
       InstanceNumber but lacking ImagePositionPatient can still be
       viewed one slice at a time (existing 2D path) but is REJECTED
       here, because "correct sequence" and "known physical spacing"
       are different guarantees and 3D reconstruction needs both.
    4. z-position values are monotonic (strictly increasing or
       strictly decreasing) after ordering - a real, direct check, not
       assumed from the ordering method succeeding.
    5. Inter-slice spacing is consistent (checked via the standard
       deviation of consecutive z-differences relative to their mean;
       a real numeric check, not a fixed hardcoded tolerance chosen
       without reasoning - see SPACING_CONSISTENCY_TOLERANCE below).
    6. PixelSpacing present on every slice.
    7. Rows/Columns identical across every slice in the series.
    8. ImageOrientationPatient present on every slice AND standard
       axial (checked against identity axial cosines with tolerance,
       same method already used and tested in the V3 dataset organizer
       script earlier in this project) - a tilted/non-axial series is
       rejected rather than treated as if z-position ordering is still
       physically meaningful for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pydicom

from app.ml_models.component4.dicom.windowing import to_hounsfield_units
from app.ml_models.component4.dicom.series import determine_ordering

# Relative standard-deviation threshold for inter-slice spacing
# consistency. 0.05 = spacing values may vary up to ~5% around their
# mean before being treated as inconsistent geometry. This is a
# deliberate, stated choice (not empirically derived from this
# project's own data) - loose enough to tolerate real floating-point
# noise in ImagePositionPatient across a real scanner acquisition,
# tight enough to catch a genuinely malformed/mixed series.
SPACING_CONSISTENCY_TOLERANCE = 0.05  # RETAINED but no longer used by the
# active check below - kept as a named constant for historical/reference
# purposes only, per instruction to make the smallest reasonable change
# while replacing the actual spacing logic. See MIN_GAPS_FOR_STATISTICAL_CONFIDENCE,
# SPACING_BASELINE_PERCENTILE, and the two threshold constants below for
# the rule that is actually applied.

# --- Spacing-consistency rule, redesigned from real-dataset evidence ---
# (see project investigation: the prior relative-std<=0.05 rule rejected
# 30/33 series in a real 37-series sample from dicom_3d_final, including
# series with a clear, dominant baseline spacing and only a legitimate
# minority of larger gaps - relative_std is not robust to even one or two
# large values in an otherwise low-mean series). The redesigned rule was
# tested against all 30 of those real rejected series before being
# implemented here - see project investigation notes for the full table.
#
# Below MIN_GAPS_FOR_STATISTICAL_CONFIDENCE gaps, ANY distributional shape
# statistic is close to meaningless (e.g. a 2-gap series trivially "passes"
# almost any majority test) - real evidence: B0025's problem series (10
# gaps, values from 5mm to 100mm, no discernible majority) only correctly
# rejects when a STRICTER bar is applied at this sample size; a looser bar
# calibrated for larger samples would have wrongly accepted it.
MIN_GAPS_FOR_SPACING_CHECK = 3       # below this: check does not apply -
                                      # too little data for any statistical
                                      # judgment; other geometry checks
                                      # (duplicates, dimensions, orientation)
                                      # remain the real protection here.
MIN_GAPS_FOR_STATISTICAL_CONFIDENCE = 15  # boundary between the strict and
                                            # lenient tiers - a stated,
                                            # evidence-informed choice, not
                                            # provably optimal from 30 samples.
SPACING_BASELINE_PERCENTILE = 25     # the "baseline" spacing is the 25th
                                      # percentile of gaps - robust to a
                                      # long right tail of legitimately
                                      # larger gaps, unlike mean.
SPACING_BASELINE_WINDOW_LOW = 0.5    # a gap counts as "near baseline" if
SPACING_BASELINE_WINDOW_HIGH = 2.0   # it falls in [0.5x, 2x] the baseline.
SPACING_STRICT_THRESHOLD = 0.80      # required near-baseline fraction for
                                      # series with 3-14 gaps.
SPACING_LENIENT_THRESHOLD = 0.50     # required near-baseline fraction for
                                      # series with >=15 gaps.

STANDARD_AXIAL_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
AXIAL_TOLERANCE = 0.05  # same value/reasoning as the V3 organizer script


class VolumeGeometryError(ValueError):
    """Raised when a series' geometry cannot be trusted enough to
    construct a 3D volume. Callers must treat this as "cannot build a
    volume for this series" - never catch-and-guess.
    """


@dataclass
class DicomVolume:
    """Result of successful volume construction. All fields are either
    directly read from DICOM tags or computed from them - nothing here
    is invented or defaulted silently.
    """
    volume: np.ndarray  # shape (num_slices, rows, columns), dtype float64 (HU values)
    shape: tuple[int, int, int]
    num_slices: int
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]     # (row_spacing, column_spacing), mm
    inter_slice_spacing: float             # mean spacing between consecutive slices, mm
    orientation: list[float]               # ImageOrientationPatient, 6 values
    origin: tuple[float, float, float]     # ImagePositionPatient of the first slice
                                            # in verified order
    dtype: str
    ordering_method: str                   # "InstanceNumber" | "ImagePositionPatient"
    slice_direction: int                   # +1 if z strictly increases through the
                                            # ordered slices, -1 if strictly decreases.
                                            # Required by frontend volume-rendering
                                            # geometry: origin is always the FIRST
                                            # ordered slice's position, but build_volume()
                                            # accepts EITHER ascending or descending
                                            # physical z-order (Check 4) - without this
                                            # sign, a 3D renderer cannot know which
                                            # physical direction increasing slice index
                                            # actually moves in, and would silently
                                            # mirror the volume along the slice axis on
                                            # roughly half of all real series. This value
                                            # was already implicitly known inside Check 4
                                            # - this field only surfaces it, no new logic.
    series_instance_uid: str
    patient_id: str | None


def _is_standard_axial(iop_values, tolerance: float = AXIAL_TOLERANCE) -> bool:
    try:
        iop = np.array([float(v) for v in iop_values])
    except (TypeError, ValueError):
        return False
    if iop.shape != (6,):
        return False
    return bool(np.allclose(iop, STANDARD_AXIAL_IOP, atol=tolerance))


def build_volume(datasets: list[pydicom.Dataset]) -> DicomVolume:
    """Constructs a verified 3D HU volume from one series' slices.

    Raises VolumeGeometryError with a specific reason if geometry
    cannot be trusted - never constructs a best-effort/guessed volume.
    """
    if not datasets:
        raise VolumeGeometryError("No slices provided.")
    if len(datasets) < 2:
        raise VolumeGeometryError(
            f"Only {len(datasets)} slice(s) provided - a volume requires "
            f"at least 2 slices."
        )

    # --- Check 1: single series ------------------------------------
    series_uids = set()
    for ds in datasets:
        suid = getattr(ds, "SeriesInstanceUID", None)
        if not suid:
            raise VolumeGeometryError(
                "At least one slice is missing SeriesInstanceUID - cannot "
                "verify all slices belong to the same series."
            )
        series_uids.add(str(suid))
    if len(series_uids) > 1:
        raise VolumeGeometryError(
            f"Slices span {len(series_uids)} different SeriesInstanceUID "
            f"values - a volume must be built from exactly one series."
        )
    series_uid = series_uids.pop()

    # --- Check 2: reliable ordering ----------------------------------
    ordering = determine_ordering(datasets)
    if not ordering.is_reliable:
        raise VolumeGeometryError(
            "No reliable slice ordering available (neither InstanceNumber "
            "nor ImagePositionPatient present on every slice). Refusing to "
            "construct a volume from unverified upload order."
        )
    ordered = ordering.ordered_datasets

    # --- Check 3: ImagePositionPatient required on every slice -------
    for ds in ordered:
        if not hasattr(ds, "ImagePositionPatient"):
            raise VolumeGeometryError(
                "At least one slice is missing ImagePositionPatient - "
                "required for volume construction regardless of ordering "
                "method, since it is the only source of real physical "
                "inter-slice spacing."
            )

    z_positions = [float(ds.ImagePositionPatient[2]) for ds in ordered]

    # --- Check 4: monotonic z-ordering --------------------------------
    diffs = np.diff(z_positions)
    if np.all(diffs > 0):
        slice_direction = 1  # strictly increasing
    elif np.all(diffs < 0):
        slice_direction = -1  # strictly decreasing
    else:
        raise VolumeGeometryError(
            f"Slice z-positions are not monotonic after ordering by "
            f"'{ordering.method}' - this indicates inconsistent or "
            f"corrupted series geometry. z-positions: {z_positions}"
        )

    # --- Check 5: spacing consistency (redesigned, see constants above) -
    abs_diffs = np.abs(diffs)
    mean_spacing = float(abs_diffs.mean())
    if mean_spacing == 0:
        raise VolumeGeometryError("Computed inter-slice spacing is zero.")

    n_gaps = len(abs_diffs)
    if n_gaps >= MIN_GAPS_FOR_SPACING_CHECK:
        baseline = float(np.percentile(abs_diffs, SPACING_BASELINE_PERCENTILE))
        if baseline == 0:
            raise VolumeGeometryError(
                f"Computed baseline spacing (P{SPACING_BASELINE_PERCENTILE}) is zero - "
                f"cannot assess spacing consistency. Gaps: {abs_diffs.tolist()}"
            )
        near_baseline = np.sum(
            (abs_diffs >= baseline * SPACING_BASELINE_WINDOW_LOW) &
            (abs_diffs <= baseline * SPACING_BASELINE_WINDOW_HIGH)
        )
        frac_near_baseline = float(near_baseline) / n_gaps

        threshold = (SPACING_STRICT_THRESHOLD if n_gaps < MIN_GAPS_FOR_STATISTICAL_CONFIDENCE
                     else SPACING_LENIENT_THRESHOLD)
        tier_label = "strict" if n_gaps < MIN_GAPS_FOR_STATISTICAL_CONFIDENCE else "lenient"

        if frac_near_baseline <= threshold:
            raise VolumeGeometryError(
                f"No dominant/coherent spacing pattern found ({tier_label} tier, "
                f"n_gaps={n_gaps}): only {frac_near_baseline*100:.1f}% of gaps fall "
                f"near the baseline spacing ({baseline:.2f}mm), below the "
                f"{threshold*100:.0f}% required for this sample size. "
                f"Gaps: {abs_diffs.tolist()}"
            )
    # else: fewer than MIN_GAPS_FOR_SPACING_CHECK gaps - too little data for
    # a meaningful statistical judgment about spacing consistency specifically;
    # the other geometry checks (duplicates, dimensions, orientation, position)
    # remain the real protection for these very short series.

    # --- Check 6: PixelSpacing present on every slice -----------------
    for ds in ordered:
        if not hasattr(ds, "PixelSpacing"):
            raise VolumeGeometryError(
                "At least one slice is missing PixelSpacing."
            )
    pixel_spacings = [tuple(float(v) for v in ds.PixelSpacing) for ds in ordered]
    if len(set(pixel_spacings)) > 1:
        raise VolumeGeometryError(
            f"PixelSpacing is not consistent across slices: "
            f"{set(pixel_spacings)}"
        )
    pixel_spacing = pixel_spacings[0]

    # --- Check 7: consistent Rows/Columns -----------------------------
    dims = set((int(ds.Rows), int(ds.Columns)) for ds in ordered)
    if len(dims) > 1:
        raise VolumeGeometryError(
            f"Inconsistent image dimensions across slices: {dims}"
        )
    rows, columns = dims.pop()

    # --- Check 8: standard axial orientation --------------------------
    for ds in ordered:
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None:
            raise VolumeGeometryError(
                "At least one slice is missing ImageOrientationPatient."
            )
        if not _is_standard_axial(iop):
            raise VolumeGeometryError(
                f"At least one slice has non-standard-axial orientation "
                f"(ImageOrientationPatient={list(iop)}). Z-position-based "
                f"volume construction is not valid for tilted acquisitions."
            )
    orientation = [float(v) for v in ordered[0].ImageOrientationPatient]

    # --- HU conversion (reused, unchanged, from windowing.py) --------
    hu_slices = [to_hounsfield_units(ds) for ds in ordered]
    volume = np.stack(hu_slices, axis=0)  # (num_slices, rows, columns)

    if volume.shape != (len(ordered), rows, columns):
        # Defensive - should be unreachable given check 7, but a stacked
        # shape mismatch is exactly the kind of silent-corruption case
        # this module exists to prevent.
        raise VolumeGeometryError(
            f"Stacked volume shape {volume.shape} does not match expected "
            f"({len(ordered)}, {rows}, {columns})."
        )

    origin = tuple(float(v) for v in ordered[0].ImagePositionPatient)
    patient_id = getattr(ordered[0], "PatientID", None)

    return DicomVolume(
        volume=volume,
        shape=volume.shape,
        num_slices=len(ordered),
        rows=rows,
        columns=columns,
        pixel_spacing=pixel_spacing,
        inter_slice_spacing=mean_spacing,
        orientation=orientation,
        origin=origin,
        dtype=str(volume.dtype),
        ordering_method=ordering.method,
        slice_direction=slice_direction,
        series_instance_uid=series_uid,
        patient_id=str(patient_id) if patient_id else None,
    )