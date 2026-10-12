"""
combined_gradcam.py - Combined dual-branch Grad-CAM visualization.

Consumes already-computed raw DenseNet and Custom CNN heatmaps from
gradcam_core.compute_raw_heatmap() - performs no Grad-CAM computation
of its own. Metrics are calculated directly from the raw float
heatmap arrays, never from a saved/colormapped image.
"""

import os
import uuid
import numpy as np
import cv2

OUTPUT_DIR = "static/gradcam/component4/combined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTIVATION_THRESHOLD = 0.5
DENSENET_COLOR = np.array([178, 114, 0], dtype=np.float32)
CUSTOM_COLOR = np.array([0, 159, 230], dtype=np.float32)


def _compute_metrics(dense_hm_224: np.ndarray, custom_hm_224: np.ndarray) -> dict:
    """Unchanged from the original - same formulas, same threshold."""
    dense_mask = dense_hm_224 >= ACTIVATION_THRESHOLD
    custom_mask = custom_hm_224 >= ACTIVATION_THRESHOLD
    intersection = np.logical_and(dense_mask, custom_mask).sum()
    union = np.logical_or(dense_mask, custom_mask).sum()
    iou = float(intersection / union) if union > 0 else 0.0
    dense_area = float(dense_mask.mean())
    custom_area = float(custom_mask.mean())

    def center_of_mass(hm):
        ys, xs = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
        total = hm.sum()
        if total == 0:
            return (hm.shape[0] / 2, hm.shape[1] / 2)
        return (float((ys * hm).sum() / total), float((xs * hm).sum() / total))

    dense_com = center_of_mass(dense_hm_224)
    custom_com = center_of_mass(custom_hm_224)
    com_distance = float(np.sqrt((dense_com[0] - custom_com[0]) ** 2 + (dense_com[1] - custom_com[1]) ** 2))

    return {
        "iou_at_0.5": iou,
        "dense_area_above_0.5": dense_area,
        "custom_area_above_0.5": custom_area,
        "center_of_mass_distance_px": com_distance,
    }


def generate_combined_gradcam(
    image_bytes: bytes,
    dense_raw: np.ndarray,
    dense_idx: int,
    custom_raw: np.ndarray,
    custom_idx: int,
    show_contours: bool = True,
    filename_tag: str = "combined",
):
    """REFACTORED signature: dense_raw/dense_idx/custom_raw/custom_idx
    are now REQUIRED parameters - the two already-computed heatmaps and
    predicted indices, NOT recomputed here. The caller (run_prediction())
    is responsible for computing each exactly once and passing the
    result to every function that needs it.

    Same defensive same-class check as the original, same visual
    output, same metrics.
    """
    if dense_idx != custom_idx:
        raise RuntimeError(
            f"Predicted class mismatch between branches: DenseNet={dense_idx}, "
            f"Custom={custom_idx}. This should not happen - do not proceed "
            f"with a combined visualization until this is investigated."
        )

    dense_224 = cv2.resize(dense_raw, (224, 224))
    custom_224 = cv2.resize(custom_raw, (224, 224))

    nparr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.resize(original, (224, 224)).astype(np.float32)

    base = original * 0.5
    dense_contrib = DENSENET_COLOR[None, None, :] * dense_224[..., None]
    custom_contrib = CUSTOM_COLOR[None, None, :] * custom_224[..., None]
    combined = base + dense_contrib * 0.6 + custom_contrib * 0.6
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    if show_contours:
        dense_mask_u8 = (dense_224 >= ACTIVATION_THRESHOLD).astype(np.uint8) * 255
        custom_mask_u8 = (custom_224 >= ACTIVATION_THRESHOLD).astype(np.uint8) * 255
        dense_contours, _ = cv2.findContours(dense_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        custom_contours, _ = cv2.findContours(custom_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined, dense_contours, -1, tuple(int(c) for c in DENSENET_COLOR), 1)
        cv2.drawContours(combined, custom_contours, -1, tuple(int(c) for c in CUSTOM_COLOR), 1)

    metrics = _compute_metrics(dense_224, custom_224)
    metrics["predicted_class_index"] = dense_idx

    filename = f"{uuid.uuid4()}_{filename_tag}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, combined)
    combined_url = f"/static/gradcam/component4/combined/{filename}"

    return combined_url, metrics, combined