"""
Renders a windowed DICOM slice into PNG bytes in exactly the format
the EXISTING inference.preprocess() and gradcam.generate_gradcam()
functions already expect (cv2.imdecode-compatible bytes), so the
windowed slice can flow through the unmodified, already-tested
classification and Grad-CAM code paths without a parallel DICOM-specific
inference implementation.

Deliberately isolated in its own module (per the requirement that this
transformation stay easy to revise later) — if preprocessing-parity
validation against real cases later shows a mismatch with the model's
training distribution, only this file should need to change.

IMPORTANT DOCUMENTED LIMITATION: the exact preprocessing used to
prepare the model's original training images is unknown (no training
script was available for inspection). The scaling approach below
(min-max scale the windowed HU range to 0-255, replicate grayscale to
3 channels) is a reasonable, defensible default for CT windowing
exports — it is NOT confirmed to match the training distribution.
Treat classifier output on DICOM-derived input with the same
"research prototype, not a validated clinical pipeline" caveat that
already applies to the existing PNG workflow, until empirical
parity-testing is done (see project notes).
"""

from __future__ import annotations

import cv2
import numpy as np
import pydicom

from app.ml_models.component4.dicom.windowing import apply_window, to_hounsfield_units


def render_slice_to_png_bytes(
    dataset: pydicom.Dataset,
    window_center: float,
    window_width: float,
) -> bytes:
    """DICOM dataset -> HU -> windowed -> 8-bit -> 3-channel -> PNG bytes."""
    hu_array = to_hounsfield_units(dataset)
    windowed = apply_window(hu_array, window_center, window_width)  # uint8, 1-channel

    # Replicate grayscale to 3 channels to match the shape cv2.imdecode
    # would produce for the existing PNG/JPG upload path (IMREAD_COLOR).
    three_channel = cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)

    success, encoded = cv2.imencode(".png", three_channel)
    if not success:
        raise ValueError("Failed to encode rendered DICOM slice to PNG.")

    return encoded.tobytes()


def render_slice_preview(
    dataset: pydicom.Dataset,
    window_center: float,
    window_width: float,
) -> bytes:
    """Same rendering, exposed as a distinct function name for the
    viewer's slice-preview endpoint. Currently identical to
    render_slice_to_png_bytes, but kept as a separate entry point since
    the viewer preview and the model-input rendering are conceptually
    different consumers (per the "display image vs model image" — they
    happen to be the same today, but should not be assumed to always
    be the same in the future, e.g. if the viewer preview later gains
    overlays/annotations that must NOT leak into what the model sees).
    """
    return render_slice_to_png_bytes(dataset, window_center, window_width)