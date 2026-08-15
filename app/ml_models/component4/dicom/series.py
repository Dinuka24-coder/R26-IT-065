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