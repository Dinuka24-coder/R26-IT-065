import base64
import os
import uuid
import cv2
import numpy as np

from app.ml_models.component4.model import get_model
from app.ml_models.component4.gradcam_core import compute_raw_heatmap


HEATMAP_DIR = "static/gradcam/component4"
os.makedirs(HEATMAP_DIR, exist_ok=True)


def encode_png_data_uri(image: np.ndarray) -> str:
    """Encodes a BGR uint8 image as an in-memory PNG data URI
    ("data:image/png;base64,..."). No file is created."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode Grad-CAM image as PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def generate_gradcam(
    image_bytes: bytes,
    model=None,
    layer_name: str = "texture_conv_4",
    return_heatmap_only: bool = False,
    filename_tag: str = "",
    precomputed_heatmap=None,
    as_data_uri: bool = False,
):
    """Returns a str (the overlay URL) by default - EXACT existing
    behavior for every current caller (both PNG and DICOM), unchanged.

    return_heatmap_only=True is a NEW, opt-in-only parameter: when set,
    ALSO saves the raw heatmap_color image (the colormap BEFORE
    blending with the original - already computed below, previously
    always discarded after compositing) as a second file, and returns
    a (overlay_url, heatmap_only_url) tuple instead of a bare string.

    Default remains False specifically so every existing call site -
    both the PNG path (until updated) and the DICOM path
    (comp4_service.py's generate_gradcam(rendered_png_bytes,
    model=get_dicom_model())), which is NOT touched by this change -
    continues receiving exactly the same str return value as before.

    filename_tag (NEW - this was previously called by comp4_service.py
    but did not yet exist as a parameter here; adding it now fixes a
    real TypeError that would have occurred on any /predict call):
    optional, purely additive, default "" - when non-empty, inserted
    into the saved filename (e.g. "densenet" ->
    "{uuid}_densenet_gradcam.png"). Every call that omits it keeps the
    original "{uuid}_gradcam.png" pattern exactly.

    precomputed_heatmap: optional (raw_heatmap, predicted_index,
    predictions) tuple - when provided, the GradientTape computation
    (now in gradcam_core.compute_raw_heatmap()) is SKIPPED entirely and
    this exact result is used for rendering. Every existing caller that
    omits this argument computes fresh, exactly as before.

    as_data_uri (NEW, opt-in, default False): when True, NOTHING is
    written to disk. The overlay is returned as an in-memory PNG data
    URI string instead of a "/static/..." URL; with return_heatmap_only
    =True the raw heatmap is returned the same way, as the second tuple
    element. The image content is byte-for-byte what would have been
    saved (same cv2 PNG encoding). Every existing caller omits it, so
    the DICOM path keeps writing files and receiving a URL, unchanged.
    """
    if model is None:
        model = get_model()

    if precomputed_heatmap is not None:
        heatmap, predicted_index, predictions = precomputed_heatmap
    else:
        heatmap, predicted_index, predictions = compute_raw_heatmap(image_bytes, model, layer_name)

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    nparr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original is None:
        raise ValueError("Invalid image file for Grad-CAM")

    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    if as_data_uri:
        overlay_data_uri = encode_png_data_uri(overlay)
        if not return_heatmap_only:
            return overlay_data_uri
        return overlay_data_uri, encode_png_data_uri(heatmap_color)

    filename = f"{uuid.uuid4()}_{filename_tag + '_' if filename_tag else ''}gradcam.png"
    save_path = os.path.join(HEATMAP_DIR, filename)

    cv2.imwrite(save_path, overlay)

    overlay_url = f"/static/gradcam/component4/{filename}"

    if not return_heatmap_only:
        return overlay_url

    # heatmap_color was already computed above for the overlay blend -
    # this just saves it as its own file too, rather than discarding it.
    # Same uuid stem as the overlay, distinct suffix, so both files from
    # one analysis are identifiable as a pair.
    heatmap_filename = filename.replace("_gradcam.png", "_gradcam_heatmap.png")
    heatmap_save_path = os.path.join(HEATMAP_DIR, heatmap_filename)
    cv2.imwrite(heatmap_save_path, heatmap_color)
    heatmap_only_url = f"/static/gradcam/component4/{heatmap_filename}"

    return overlay_url, heatmap_only_url