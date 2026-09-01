import numpy as np
import cv2
import tensorflow as tf
import base64
from app.ml_models.component1.model import get_model

IMG_SIZE  = (224, 224)
THRESHOLD = 0.35   # remove weak activations below this

# ── U-Net pleural line enhancement ─────────────────────────────
# Disabled: the current U-Net reaches only Dice 0.40, so the drawn
# pleural line is unreliable and can mislead interpretation.
# Re-enable once Focal Tversky training improves segmentation.
# The weights are imported lazily inside get_pleural_segmentation(),
# so nothing is loaded into memory while this is False.
USE_UNET_ENHANCEMENT = False

# Set False to silence the per-stage terminal output
VERBOSE = True


def _log(msg):
    if VERBOSE:
        print(msg)


def preprocess_for_gradcam(image_bytes: bytes) -> tuple:
    """Preprocess image and return model input + original RGB"""
    nparr = np.frombuffer(image_bytes, np.uint8)

    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_original = cv2.resize(img_original, IMG_SIZE)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img   = cv2.resize(img, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    img_model = np.stack([img_clahe, img_clahe, img_clahe], axis=-1).astype(np.float32)

    return np.expand_dims(img_model, axis=0), img_original, img


def generate_standard_gradcam_overlay(image_bytes: bytes):
    """Standard Grad-CAM WITHOUT lung masking or boundary — for comparison"""
    _log("   📊 STANDARD Grad-CAM (comparison overlay)")

    model = get_model()
    img_batch, img_original, gray_img = preprocess_for_gradcam(image_bytes)
    efficientnet = model.layers[1]

    grad_model = tf.keras.Model(
        inputs  = efficientnet.input,
        outputs = [efficientnet.get_layer("top_activation").output, efficientnet.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, effnet_output = grad_model(img_batch)
        tape.watch(conv_outputs)
        x = effnet_output
        for layer in model.layers[2:]:
            x = layer(x)
        loss = x[:, 0]

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

    # NO masking, NO thresholding, NO boundary — raw standard Grad-CAM
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_original, 0.6, heatmap_colored, 0.4, 0)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer   = cv2.imencode(".png", overlay_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def segment_lung_region(gray_img: np.ndarray) -> np.ndarray:
    """
    Create a lung region mask using a combination of:
    1. Central anatomical region (lungs are in the middle of the chest)
    2. Intensity filtering (exclude very bright bone and very dark background)
    """
    h, w = gray_img.shape

    # ── Step 1: Central chest region mask ──────────────────────
    central_mask = np.zeros((h, w), np.float32)

    x_start = int(w * 0.15)   # exclude 15% left edge (arm/shoulder)
    x_end   = int(w * 0.85)   # exclude 15% right edge (arm/shoulder)
    y_start = int(h * 0.12)   # exclude top 12% (neck/clavicle)
    y_end   = int(h * 0.78)   # exclude bottom 22% (abdomen)

    central_mask[y_start:y_end, x_start:x_end] = 1.0

    # ── Step 2: Intensity filtering ────────────────────────────
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray_img)
    img_norm = img_eq.astype(np.float32) / 255.0

    intensity_mask = np.ones((h, w), np.float32)
    intensity_mask[img_norm < 0.08] = 0.0   # too dark (background)
    intensity_mask[img_norm > 0.88] = 0.3   # too bright (bone) - reduce not remove

    # ── Step 3: Combine both masks ─────────────────────────────
    lung_mask = central_mask * intensity_mask

    # ── Step 4: Smooth the mask edges ──────────────────────────
    lung_mask = cv2.GaussianBlur(lung_mask, (31, 31), 0)

    if lung_mask.max() > 0:
        lung_mask = lung_mask / lung_mask.max()

    return lung_mask


def get_pleural_segmentation(gray_full_img: np.ndarray) -> tuple:
    """
    Use trained U-Net to segment the exact pneumothorax region
    and extract the precise pleural line.

    Only called when USE_UNET_ENHANCEMENT is True — the import is
    local so the weights are never loaded while it is disabled.
    """
    from app.ml_models.component1.unet_model import get_unet_model
    unet = get_unet_model()

    img_256 = cv2.resize(gray_full_img, (256, 256))
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_256 = clahe.apply(img_256).astype(np.float32) / 255.0
    img_batch = img_256[np.newaxis, ..., np.newaxis]

    pred_mask = unet.predict(img_batch, verbose=0)[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (224, 224))

    kernel    = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pleural_line = np.zeros((224, 224), np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # ignore tiny specks
            cv2.drawContours(pleural_line, [cnt], -1, 255, 2)

    return pred_mask, pleural_line


def generate_boundary_aware_gradcam(image_bytes: bytes) -> dict:
    """
    Boundary-Aware Grad-CAM:
    1. Standard Grad-CAM
    2. Lung region masking (constrains to lung only)
    3. Thresholding (removes weak activations)
    4. Boundary extraction (Canny edge on pleural line)
    5. Negative space detection (the void)

    Note: Grad-CAM is ONLY generated when Pneumothorax is detected.
    Normal / Not-Pneumothorax cases skip all heatmap work.
    """
    _log("\n🎯 BOUNDARY-AWARE GRAD-CAM")

    model = get_model()
    img_batch, img_original, gray_img = preprocess_for_gradcam(image_bytes)

    # ── PREDICT FIRST ──────────────────────────────────────────
    raw_score  = float(model.predict(img_batch, verbose=0)[0][0])
    label      = "Pneumothorax Detected" if raw_score >= 0.5 else "No Pneumothorax"
    confidence = round(raw_score * 100, 2) if raw_score >= 0.5 \
                 else round((1 - raw_score) * 100, 2)

    # ── EARLY RETURN: skip Grad-CAM for normal cases ───────────
    if raw_score < 0.5:
        _log(f"   → NEGATIVE (score {raw_score:.4f}, {confidence}% confidence)")
        _log("   → pipeline skipped — no heatmap generated for normal cases")
        return {
            "prediction":          label,
            "confidence":          confidence,
            "raw_score":           round(raw_score, 4),
            "affected_lung_pct":   0.0,
            "boundary_length_pct": 0.0,
            "pleural_separation":  False,
            "segmented_area_pct":  0.0,
            "heatmap_base64":      None,
        }

    _log(f"   → POSITIVE (score {raw_score:.4f}, {confidence}% confidence)")
    _log("   → running 4-stage boundary-aware pipeline")

    # ═══ ONLY RUNS IF PNEUMOTHORAX DETECTED ════════════════════
    efficientnet = model.layers[1]

    # ── STEP 1: Standard Grad-CAM ──────────────────────────────
    grad_model = tf.keras.Model(
        inputs  = efficientnet.input,
        outputs = [
            efficientnet.get_layer("top_activation").output,
            efficientnet.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, effnet_output = grad_model(img_batch)
        tape.watch(conv_outputs)
        x = effnet_output
        for layer in model.layers[2:]:
            x = layer(x)
        pred = x
        loss = pred[:, 0]

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    _log(f"   [1] Grad-CAM        {heatmap.shape[0]}x{heatmap.shape[1]} → 224x224, "
         f"max activation {heatmap_resized.max():.3f}")

    # ── STEP 2: Lung Region Masking ────────────────────────────
    lung_mask = segment_lung_region(gray_img)
    lung_px   = int(np.sum(lung_mask > 0.3))
    before_px = int(np.sum(heatmap_resized > THRESHOLD))

    heatmap_masked = heatmap_resized * lung_mask
    if heatmap_masked.max() > 0:
        heatmap_masked = heatmap_masked / heatmap_masked.max()

    after_px = int(np.sum(heatmap_masked > THRESHOLD))
    _log(f"   [2] Lung masking    {lung_px} px lung field · "
         f"activation {before_px} → {after_px} px "
         f"({100*(1-after_px/before_px):.0f}% removed)" if before_px else
         f"   [2] Lung masking    {lung_px} px lung field")

    # ── STEP 3: Thresholding ───────────────────────────────────
    heatmap_thresh = heatmap_masked.copy()
    heatmap_thresh[heatmap_thresh < THRESHOLD] = 0
    kept_px = int(np.sum(heatmap_thresh > THRESHOLD))
    _log(f"   [3] Threshold {THRESHOLD}   {kept_px} px retained above threshold")

    # ── STEP 4: Boundary Extraction (Canny) ────────────────────
    heatmap_uint8 = np.uint8(255 * heatmap_thresh)
    edges = cv2.Canny(heatmap_uint8, 30, 100)

    kernel        = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edge_px = int(np.sum(edges_dilated > 0))
    _log(f"   [4] Canny boundary  {edge_px} edge px extracted (drawn cyan)")

    # ── STEP 5: Negative Space Detection ───────────────────────
    lung_binary     = (lung_mask > 0.3).astype(np.uint8) * 255
    high_activation = (heatmap_thresh > THRESHOLD).astype(np.uint8) * 255

    negative_space = cv2.bitwise_and(
        lung_binary,
        cv2.bitwise_not(high_activation)
    )
    kernel_ns      = np.ones((5, 5), np.uint8)
    negative_space = cv2.morphologyEx(negative_space, cv2.MORPH_OPEN, kernel_ns)
    _log(f"   [5] Negative space  {int(np.sum(negative_space > 0))} px void detected")

    # ── STEP 6: Build the Clean Overlay ────────────────────────
    overlay = img_original.copy().astype(np.float32)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_thresh), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32)

    activation_mask = (heatmap_thresh > 0)[..., np.newaxis]
    overlay = np.where(
        activation_mask,
        overlay * 0.55 + heatmap_colored * 0.45,
        overlay
    )

    # Layer 2: Boundary line in bright CYAN
    overlay[edges_dilated > 0] = [0, 255, 255]

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # ── STEP 7: Metrics ────────────────────────────────────────
    lung_area       = int(np.sum(lung_mask > 0.3))
    activation_area = int(np.sum(heatmap_thresh > THRESHOLD))
    boundary_pixels = int(np.sum(edges_dilated > 0))

    affected_pct = round((activation_area / lung_area) * 100, 2) if lung_area > 0 else 0.0
    boundary_length_pct = round((boundary_pixels / (IMG_SIZE[0] * IMG_SIZE[1])) * 100, 2)

    _log(f"   [M] Metrics         affected {affected_pct}% · "
         f"boundary {boundary_length_pct}% · "
         f"separation {'PRESENT' if boundary_length_pct > 1.0 else 'ABSENT'}")

    # ── STEP 8: Convert to base64 ──────────────────────────────
    overlay_bgr    = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer      = cv2.imencode(".png", overlay_bgr)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # ── STEP 9: Precise Pleural Line via U-Net (optional) ──────
    seg_area_pct = 0.0

    if USE_UNET_ENHANCEMENT:
        try:
            seg_mask, pleural_line = get_pleural_segmentation(gray_img)

            overlay[pleural_line > 0] = [255, 255, 0]

            overlay_bgr    = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buffer      = cv2.imencode(".png", overlay_bgr)
            overlay_base64 = base64.b64encode(buffer).decode("utf-8")

            seg_area_pct = round((int(np.sum(seg_mask > 0)) / (224 * 224)) * 100, 2)
            _log("   [6] U-Net           pleural line drawn (yellow)")

        except Exception as e:
            _log(f"   [6] U-Net           skipped: {e}")
            seg_area_pct = 0.0
    else:
        _log("   [6] U-Net           disabled (Dice 0.40 — pending improvement)")

    # ── Generate Standard Grad-CAM for comparison ──────────────
    standard_heatmap = generate_standard_gradcam_overlay(image_bytes)

    _log("   ✅ boundary-aware pipeline complete\n")

    return {
        "prediction":              label,
        "confidence":              float(confidence),
        "raw_score":               round(raw_score, 4),
        "affected_lung_pct":       float(affected_pct),
        "boundary_length_pct":     float(boundary_length_pct),
        "pleural_separation":      bool(boundary_length_pct > 1.0),
        "segmented_area_pct":      float(seg_area_pct),
        "heatmap_base64":          overlay_base64,
        "standard_heatmap_base64": standard_heatmap,
    }