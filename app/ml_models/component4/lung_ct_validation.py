"""
Level 2 validation: best-effort lung CT suitability check.

CRITICAL FRAMING — read before touching this file:

This is NOT a validated diagnostic or anatomical classifier. It is a
lightweight, best-effort filter intended to catch obviously unsuitable
input (natural photos, blank/corrupt images, wildly wrong aspect
ratios) before it reaches the cancer classifier. It cannot reliably
confirm that an image is specifically a *lung* CT slice as opposed to,
say, a brain or abdominal CT, or even a chest X-ray in some cases —
that level of reliability would require a dedicated trained binary
classifier (Lung CT vs Not Lung CT), which does not exist yet. See
"FUTURE EXTENSION POINT" below.

check_lung_ct_suitability() returns a ValidationResult, never a bare
bool, specifically so the failure reason is always available for
logging/debugging without needing to re-derive it.

A validation failure here means exactly one thing:

    "This input is unsuitable for this component."

It must NEVER be interpreted, displayed, or logged as:

    "This patient does not have cancer."

Those are unrelated claims. Callers (comp4_service.py) must map a
failed ValidationResult to a rejection response, not to a "Normal"
classification result.

This function runs identically for:
  - PNG/JPG uploads, after existing file-type validation, before
    inference.predict()
  - DICOM-rendered slices, after windowing/rendering, before
    inference.predict()

It operates purely on 2D rendered image bytes — it does not know or
care whether the origin was a DICOM slice or a direct PNG upload. That
symmetry is intentional: the same gate applies to both input types.

FUTURE EXTENSION POINT:
A dedicated binary classifier (Lung CT vs Not Lung CT) could replace
the body of check_lung_ct_suitability() without requiring changes to
callers, as long as it keeps the same signature
(bytes -> ValidationResult). Suggested approach when that data becomes
available: a lightweight fine-tuned classifier (e.g. MobileNetV2),
positive class from the existing 4-class lung CT dataset, negative
class assembled from this project's own sibling components' chest
X-ray data (components 1-3) plus a small sample of non-lung CT and
generic photos. Not implemented in this phase — no negative-class
dataset currently exists to train it on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --- Tunable thresholds -----------------------------------------------
# These are coarse, hand-picked starting points, not calibrated against
# a labeled dataset. Expect to revisit them once real rejected/accepted
# examples are observed in practice.

MIN_INTENSITY_STD = 8.0
"""Reject near-blank/uniform images (e.g. a solid-color or corrupt file).
Grayscale CT slices have substantial contrast; a near-flat image is a
strong signal something is wrong, independent of anatomy."""

MAX_COLOR_SATURATION = 18.0
"""Mean absolute difference between color channels. CT images and their
grayscale-window renders are effectively colorless; a high value strongly
suggests a natural color photo rather than a CT-derived image. This is a
much weaker signal for a scanned/exported color-annotated CT image, which
is a known gap."""

MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
"""Reject grossly non-square-ish images. Real CT slices are typically
square or near-square (e.g. 512x512); this is a loose sanity check, not
a precise anatomical constraint."""
# ------------------------------------------------------------------------


@dataclass
class ValidationResult:
    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    @property
    def user_message(self) -> str:
        if self.is_valid:
            return ""
        return (
            "Unsupported image. This component accepts lung CT images only. "
            "Input is unsuitable for this component."
        )


def check_lung_ct_suitability(image_bytes: bytes) -> ValidationResult:
    """Best-effort suitability check on a rendered 2D image (PNG/JPG
    upload bytes, or a DICOM-rendered slice's PNG bytes — same code
    path for both, see module docstring).
    """
    checks: dict = {}
    reasons: list[str] = []

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return ValidationResult(
            is_valid=False,
            reasons=["Image could not be decoded."],
            checks=checks,
        )

    height, width = img.shape[:2]
    aspect_ratio = width / height if height else 0
    checks["aspect_ratio"] = round(aspect_ratio, 3)
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        reasons.append("Image dimensions are not consistent with a CT slice.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    intensity_std = float(np.std(gray))
    checks["intensity_std"] = round(intensity_std, 3)
    if intensity_std < MIN_INTENSITY_STD:
        reasons.append("Image appears blank or near-uniform.")

    b, g, r = cv2.split(img.astype(np.int16))
    saturation_signal = float(
        (np.abs(b - g) + np.abs(g - r) + np.abs(b - r)).mean()
    )
    checks["color_saturation_signal"] = round(saturation_signal, 3)
    if saturation_signal > MAX_COLOR_SATURATION:
        reasons.append(
            "Image appears to be a color photo rather than a grayscale "
            "CT-derived image."
        )

    return ValidationResult(
        is_valid=len(reasons) == 0,
        reasons=reasons,
        checks=checks,
    )