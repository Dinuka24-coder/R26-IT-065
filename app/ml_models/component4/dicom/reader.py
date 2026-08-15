"""
DICOM reading and structural (file-level) validation.

This module answers exactly one question: "is this a parseable DICOM
object with usable pixel data?" It does NOT check whether it's a CT,
whether it's the right anatomical region, or whether it's suitable for
the lung cancer classifier. Those are separate, later stages (see
lung_ct_validation.py and windowing.py) — kept separate deliberately
per the project's staged-validation requirement:

    1. File validation          <- this module
    2. DICOM validation         <- this module
    3. CT modality validation   <- validation.py (in this package)
    4. Lung CT suitability      <- app.ml_models.component4.lung_ct_validation
    5. Cancer classification    <- app.ml_models.component4.inference
    6. Grad-CAM explanation     <- app.ml_models.component4.gradcam

Do not fold these stages back together.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pydicom
from pydicom.errors import InvalidDicomError


class DicomStructureError(ValueError):
    """Raised when a file is not a usable DICOM object.

    Callers should catch this and turn it into a clean, non-technical
    HTTP error for the user — never surface the raw exception/stack
    trace to the frontend.
    """


@dataclass
class ParsedDicom:
    dataset: pydicom.Dataset
    raw_bytes: bytes


def read_dicom(file_bytes: bytes) -> ParsedDicom:
    """Parse raw bytes into a pydicom Dataset and confirm it has usable
    pixel data. Raises DicomStructureError with a clean message on any
    failure — never lets a raw pydicom/struct exception escape.
    """
    if not file_bytes:
        raise DicomStructureError("Uploaded file is empty.")

    try:
        dataset = pydicom.dcmread(io.BytesIO(file_bytes), force=False)
    except InvalidDicomError as exc:
        raise DicomStructureError(
            "File is not a valid DICOM object."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        # pydicom can raise a variety of low-level errors (struct.error,
        # AttributeError, etc.) on malformed input. All of them mean the
        # same thing to the user: this isn't a readable DICOM file.
        raise DicomStructureError(
            "File could not be read as DICOM."
        ) from exc

    if "PixelData" not in dataset:
        raise DicomStructureError(
            "DICOM file does not contain pixel data."
        )

    try:
        # Accessing pixel_array forces decompression/decoding; this is
        # the real test of whether the pixel data is usable, not just
        # present.
        _ = dataset.pixel_array
    except Exception as exc:  # noqa: BLE001
        raise DicomStructureError(
            "DICOM pixel data could not be decoded. The file may use an "
            "unsupported compressed transfer syntax."
        ) from exc

    rows = getattr(dataset, "Rows", None)
    columns = getattr(dataset, "Columns", None)
    if not rows or not columns:
        raise DicomStructureError("DICOM file is missing valid image dimensions.")

    return ParsedDicom(dataset=dataset, raw_bytes=file_bytes)