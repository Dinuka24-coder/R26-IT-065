"""
Segmentation extension point for Component 4.

CRITICAL FRAMING:

There is no trained segmentation model in this project. This module does
NOT produce a segmentation mask. It exists so the API response and the
frontend UI have a stable, honest place to report that status, instead
of either silently omitting the feature or faking a result.

Do NOT threshold the Grad-CAM attention map and present it as a
segmentation mask here or anywhere else. Grad-CAM answers "what did the
classifier attend to"; segmentation answers "which pixels are the
lesion" - these require different evidence (Grad-CAM needs only the
classifier; a trustworthy segmentation mask needs pixel-level annotated
training data, which this project's DICOM dataset does not currently
have) and conflating them produces a confident-looking but meaningless
result. See the project's Grad-CAM terminology rules for the same
principle applied to the classification explanation.

FUTURE EXTENSION POINT: when a trained segmentation model (e.g. U-Net)
and its weights file become available, load it in a sibling module
(e.g. segmentation_model.py, mirroring model.py's get_model() pattern)
and update run_segmentation() below to return a real mask - keeping the
same return shape so callers (comp4_service.py) don't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SegmentationResult:
    available: bool
    reason: str = ""
    mask_url: str | None = None
    overlay_url: str | None = None


def run_segmentation(image_bytes: bytes) -> SegmentationResult:
    """Called with the same rendered slice bytes used for classification
    and Grad-CAM (see comp4_service.py) - kept as a real function call,
    not a hardcoded response, so wiring in a real model later is a
    one-function change, not a callers-wide change.
    """
    return SegmentationResult(
        available=False,
        reason=(
            "No trained segmentation model is currently available for "
            "this component. Classification and Grad-CAM attention "
            "mapping are available; pixel-level segmentation requires "
            "annotated training data this project does not yet have."
        ),
    )