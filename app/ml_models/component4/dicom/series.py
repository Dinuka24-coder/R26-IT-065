"""
Multi-file DICOM series handling and short-lived server-side cache.

The viewer workflow is inherently multi-step (inspect -> scroll slices
-> adjust window/level -> analyze), so the backend needs to hold onto
the parsed DICOM data between requests. This module is an in-memory
cache keyed by a server-generated series_id (uuid4 — never derived from
PHI or the DICOM's own UIDs, which could be identifying).

Known limitation, stated plainly: this cache is in-process memory only.
It does not survive a server restart and does not scale across multiple
worker processes. That's an acceptable tradeoff for a research
prototype with a single backend process; it would need to move to a
shared store (Redis, disk-backed temp files, etc.) before any multi-worker
or production deployment. Documenting this now rather than pretending
otherwise.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

import pydicom

from app.ml_models.component4.dicom.reader import ParsedDicom

# Series are dropped after this many seconds of inactivity.
SERIES_TTL_SECONDS = 30 * 60  # 30 minutes


@dataclass
class DicomSeries:
    series_id: str
    datasets: list[pydicom.Dataset]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def number_of_slices(self) -> int:
        return len(self.datasets)


class SeriesStore:
    """Thread-safe in-memory store for parsed DICOM series."""

    def __init__(self) -> None:
        self._series: dict[str, DicomSeries] = {}
        self._lock = Lock()

    def create(self, parsed_files: list[ParsedDicom]) -> DicomSeries:
        """Group parsed DICOM files into one series, sorted into a
        stable slice order, and cache it under a new series_id.

        Sort order preference:
          1. InstanceNumber, if present on all datasets (most reliable
             for typical axial CT series)
          2. ImagePositionPatient z-coordinate, if present
          3. Upload order, as a last resort (logged as a limitation,
             not silently treated as correct)
        """
        datasets = [pf.dataset for pf in parsed_files]
        datasets = _sort_slices(datasets)

        series_id = str(uuid.uuid4())
        series = DicomSeries(series_id=series_id, datasets=datasets)

        with self._lock:
            self._sweep_expired_locked()
            self._series[series_id] = series

        return series

    def get(self, series_id: str) -> DicomSeries | None:
        with self._lock:
            self._sweep_expired_locked()
            series = self._series.get(series_id)
            if series is not None:
                series.last_accessed = time.time()
            return series

    def _sweep_expired_locked(self) -> None:
        """Caller must hold self._lock."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._series.items()
            if now - s.last_accessed > SERIES_TTL_SECONDS
        ]
        for sid in expired:
            del self._series[sid]


def _sort_slices(datasets: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    if len(datasets) == 1:
        return datasets

    if all(hasattr(ds, "InstanceNumber") for ds in datasets):
        return sorted(datasets, key=lambda ds: int(ds.InstanceNumber))

    if all(hasattr(ds, "ImagePositionPatient") for ds in datasets):
        return sorted(datasets, key=lambda ds: float(ds.ImagePositionPatient[2]))

    # No reliable ordering tag available. Do not silently pretend upload
    # order is correct slice order for a multi-slice series — this is a
    # genuine limitation for series lacking InstanceNumber/ImagePositionPatient.
    return datasets


# Module-level singleton store, matching the existing project's
# module-level singleton pattern for the loaded Keras model (see
# app/ml_models/component4/model.py's `get_model()`).
_store = SeriesStore()


def get_series_store() -> SeriesStore:
    return _store


# ---------------------------------------------------------------------
# ADDITIVE ONLY - added for Day 3 Phase 1 (3D volume construction).
# Everything above this line is completely unchanged. _sort_slices(),
# SeriesStore, DicomSeries, and get_series_store() are not modified in
# any way - this new function exists in parallel for callers (volume.py)
# that need to know WHICH ordering method succeeded and whether it's
# reliable enough for 3D volume construction, information _sort_slices()
# deliberately doesn't expose (it silently falls back to upload order,
# which is acceptable for single-slice-at-a-time 2D viewing but not for
# building a spatially correct volume).
# ---------------------------------------------------------------------

@dataclass
class SliceOrderingResult:
    """Result of determine_ordering() below - NOT used by _sort_slices()
    or SeriesStore.create(), which are unchanged and unaffected by this
    addition.
    """
    ordered_datasets: list[pydicom.Dataset]
    method: str  # "InstanceNumber" | "ImagePositionPatient" | "UNRELIABLE"
    is_reliable: bool


def determine_ordering(datasets: list[pydicom.Dataset]) -> SliceOrderingResult:
    """Same priority logic as _sort_slices() above (InstanceNumber, then
    ImagePositionPatient, then give up) - but returns which method was
    used and whether it's reliable, instead of silently falling back to
    (and returning) upload order with no signal to the caller. Used by
    app/ml_models/component4/dicom/volume.py, which requires reliable
    ordering and refuses to construct a volume without it.
    """
    if len(datasets) == 1:
        return SliceOrderingResult(datasets, "InstanceNumber", True)

    if all(hasattr(ds, "InstanceNumber") for ds in datasets):
        ordered = sorted(datasets, key=lambda ds: int(ds.InstanceNumber))
        return SliceOrderingResult(ordered, "InstanceNumber", True)

    if all(hasattr(ds, "ImagePositionPatient") for ds in datasets):
        ordered = sorted(datasets, key=lambda ds: float(ds.ImagePositionPatient[2]))
        return SliceOrderingResult(ordered, "ImagePositionPatient", True)

    return SliceOrderingResult(datasets, "UNRELIABLE", False)

# ---------------------------------------------------------------------
# ADDITIVE ONLY - added for Acquisition-level grouping (approved
# architecture: acquisition grouping happens BEFORE build_volume(),
# which remains an unmodified, strict geometry validator for one
# candidate stack). Nothing above this line is changed.
# ---------------------------------------------------------------------

@dataclass
class AcquisitionGroup:
    """One candidate acquisition within a series - NOT yet validated by
    build_volume(). acquisition_number is the real DICOM AcquisitionNumber
    tag value (as a string) when every slice in the series has one, or
    None when it represents "the whole series treated as one acquisition"
    (either because AcquisitionNumber is missing, or because every slice
    already shares a single value - both cases behave identically: one
    group, containing everything).
    """
    acquisition_number: str | None
    datasets: list[pydicom.Dataset]


def group_by_acquisition(datasets: list[pydicom.Dataset]) -> list[AcquisitionGroup]:
    """Pure grouping function - partitions datasets by AcquisitionNumber.
    Does NOT call build_volume(), does NOT validate geometry, does NOT
    sort/order slices (that remains determine_ordering()/build_volume()'s
    job, run independently per group by the caller). This function only
    answers "which candidate acquisition does each slice belong to."

    Grouping rule, deliberately conservative (approved architecture):
      - If ANY slice in the series is missing AcquisitionNumber, the
        ENTIRE series is returned as ONE group (acquisition_number=None).
        Never attempts a partial split, never falls back to
        AcquisitionTime or InstanceNumber to fill the gap - matches the
        audit's finding of 42 real series with missing AcquisitionNumber,
        where guessing a split has no real evidence to support it.
      - If every slice HAS AcquisitionNumber but they all share the SAME
        single value, the entire series is returned as ONE group (using
        that real value, not None) - this is the 521-series
        single-acquisition case; behavior is unchanged from treating the
        series as one candidate stack, as it always has been.
      - Only when every slice has AcquisitionNumber AND more than one
        distinct value is present does this return multiple groups - the
        90-series real multi-acquisition case (e.g. E0003: three groups,
        "1"/"2"/"3").

    InstanceNumber gaps within a group are NEVER used to further split -
    explicitly out of scope for this function, per the approved
    architecture (a large gap is evaluated by build_volume()'s existing
    spacing-consistency check on that group, not treated as evidence of
    a hidden sub-acquisition here).
    """
    if not datasets:
        return []

    acquisition_numbers = [getattr(ds, "AcquisitionNumber", None) for ds in datasets]
    acquisition_numbers_str = [
        str(v) if v is not None else None for v in acquisition_numbers
    ]

    if any(v is None for v in acquisition_numbers_str):
        return [AcquisitionGroup(acquisition_number=None, datasets=list(datasets))]

    unique_values = set(acquisition_numbers_str)
    if len(unique_values) <= 1:
        single_value = acquisition_numbers_str[0] if acquisition_numbers_str else None
        return [AcquisitionGroup(acquisition_number=single_value, datasets=list(datasets))]

    groups: dict[str, list[pydicom.Dataset]] = {}
    for ds, acq in zip(datasets, acquisition_numbers_str):
        groups.setdefault(acq, []).append(ds)
    return [
        AcquisitionGroup(acquisition_number=acq, datasets=group_datasets)
        for acq, group_datasets in groups.items()
    ]