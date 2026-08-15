"""
Level 1 validation: DICOM/CT metadata validation.

This confirms the object is structurally a CT image. It does NOT
confirm it's a *lung* CT — a CT of the abdomen or head passes this
check just as easily. That distinction is intentional: metadata can
tell you the modality, not the anatomical content. Lung-specific
suitability is handled separately in lung_ct_validation.py (Level 2),
which operates on the rendered pixel content, not DICOM tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pydicom

# Modalities that are structurally acceptable as input to this workflow.
# Only "CT" is expected in normal use; this is intentionally a single-value
# allowlist rather than a denylist, so nothing is admitted by accident.
ACCEPTED_MODALITIES = {"CT"}


@dataclass
class CtValidationResult:
    is_valid: bool
    modality: str | None
    reasons: list[str] = field(default_factory=list)


def validate_ct_modality(dataset: pydicom.Dataset) -> CtValidationResult:
    """Level 1 check: is this DICOM object's Modality tag CT?

    This is a metadata check only. A CT DICOM of the wrong body part
    (e.g. head CT) will still pass this function — see module docstring.
    """
    reasons: list[str] = []
    modality = getattr(dataset, "Modality", None)

    if modality is None:
        reasons.append("DICOM file has no Modality tag.")
    elif modality not in ACCEPTED_MODALITIES:
        reasons.append(
            f"Unsupported modality '{modality}'. This component accepts CT "
            f"images only."
        )

    return CtValidationResult(
        is_valid=len(reasons) == 0,
        modality=modality,
        reasons=reasons,
    )


def safe_public_metadata(dataset: pydicom.Dataset) -> dict:
    """Extract ONLY the metadata fields needed for the viewer/research
    workflow. Deliberately an allowlist, not a denylist — new DICOM
    tags should never leak to the frontend by default.

    Explicitly excluded: PatientName, PatientBirthDate, PatientAddress,
    PatientID (the DICOM one — the application's own patient_id is used
    instead), InstitutionName, and any other PHI-bearing tag.
    """
    return {
        "modality": getattr(dataset, "Modality", None),
        "rows": getattr(dataset, "Rows", None),
        "columns": getattr(dataset, "Columns", None),
        "slice_thickness": _safe_float(getattr(dataset, "SliceThickness", None)),
        "pixel_spacing": _safe_float_list(getattr(dataset, "PixelSpacing", None)),
        "rescale_slope": _safe_float(getattr(dataset, "RescaleSlope", 1.0)),
        "rescale_intercept": _safe_float(getattr(dataset, "RescaleIntercept", 0.0)),
        "default_window_center": _safe_float_or_first(
            getattr(dataset, "WindowCenter", None)
        ),
        "default_window_width": _safe_float_or_first(
            getattr(dataset, "WindowWidth", None)
        ),
        "body_part_examined": getattr(dataset, "BodyPartExamined", None),
    }


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float_list(value) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except TypeError:
        return _safe_float(value)


def _safe_float_or_first(value) -> float | None:
    """WindowCenter/WindowWidth can legally be a single value or a list
    of values (multiple recommended presets) per the DICOM standard.
    Take the first if it's a list.
    """
    if value is None:
        return None
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return _safe_float(value[0]) if len(value) > 0 else None
    return _safe_float(value)