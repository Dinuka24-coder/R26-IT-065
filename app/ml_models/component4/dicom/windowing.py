"""
Hounsfield Unit conversion and window/level application.

    raw pixel data --(RescaleSlope, RescaleIntercept)--> HU
    HU --(Window Center, Window Width)--> clipped, scaled 0-255 display array

Presets below (WINDOW_PRESETS) are commonly used starting points for
CT windowing, NOT universal constants — actual optimal values vary by
scanner/protocol. The doctor can override width/center manually in the
viewer; these are defaults, not hardcoded "correct" values. This is
called out explicitly here and in the API response, not just in a
comment, per the project requirement not to present them as
universally correct.
"""

from __future__ import annotations

import numpy as np
import pydicom

# Common CT windowing presets (Window Center, Window Width), in HU.
# Source: widely used radiology convention, not derived from this
# project's own data or calibrated against this classifier's training set.
WINDOW_PRESETS = {
    "lung": {"window_center": -600.0, "window_width": 1500.0},
    "mediastinal": {"window_center": 50.0, "window_width": 350.0},
    "bone": {"window_center": 400.0, "window_width": 1800.0},
}

DEFAULT_PRESET = "lung"


def to_hounsfield_units(dataset: pydicom.Dataset) -> np.ndarray:
    """Convert raw stored pixel values to Hounsfield Units using the
    DICOM RescaleSlope/RescaleIntercept tags. Falls back to identity
    (slope=1, intercept=0) if the tags are absent, which is a
    documented limitation, not a silent assumption of correctness for
    scanners/exports that omit them.
    """
    pixel_array = dataset.pixel_array.astype(np.float64)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))

    return pixel_array * slope + intercept


def apply_window(
    hu_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Clip the HU array to [center - width/2, center + width/2] and
    linearly scale to uint8 [0, 255].
    """
    if window_width <= 0:
        raise ValueError("window_width must be positive.")

    lower = window_center - (window_width / 2.0)
    upper = window_center + (window_width / 2.0)

    clipped = np.clip(hu_array, lower, upper)
    scaled = (clipped - lower) / (upper - lower)  # -> [0, 1]
    return (scaled * 255.0).astype(np.uint8)


def resolve_window(
    preset: str | None,
    window_center: float | None,
    window_width: float | None,
    dataset: pydicom.Dataset | None = None,
) -> tuple[float, float, str]:
    """Decide which window/level to use, in priority order:

      1. Explicit window_center + window_width from the caller (doctor
         manually adjusted in the viewer)
      2. Named preset (lung/mediastinal/bone)
      3. Dataset's own WindowCenter/WindowWidth tag, if present
      4. Default preset (lung)

    Returns (center, width, source_label) so callers/logs can record
    which one was actually used.
    """
    if window_center is not None and window_width is not None:
        return float(window_center), float(window_width), "manual"

    if preset is not None:
        if preset not in WINDOW_PRESETS:
            raise ValueError(f"Unknown window preset '{preset}'.")
        p = WINDOW_PRESETS[preset]
        return p["window_center"], p["window_width"], f"preset:{preset}"

    if dataset is not None:
        wc = getattr(dataset, "WindowCenter", None)
        ww = getattr(dataset, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = wc[0] if isinstance(wc, (list, pydicom.multival.MultiValue)) else wc
            ww = ww[0] if isinstance(ww, (list, pydicom.multival.MultiValue)) else ww
            return float(wc), float(ww), "dataset_default"

    default = WINDOW_PRESETS[DEFAULT_PRESET]
    return default["window_center"], default["window_width"], f"preset:{DEFAULT_PRESET}"