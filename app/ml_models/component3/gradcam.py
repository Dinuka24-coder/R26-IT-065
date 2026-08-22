import cv2
import numpy as np
import tensorflow as tf
import base64


def clip_and_order_bbox(box):
    """Clips raw bbox_output coords to [0, 1] and orders them so
    x_min <= x_max and y_min <= y_max. Returns (x_min, y_min, x_max, y_max)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x_min, x_max = sorted((np.clip(x0, 0.0, 1.0), np.clip(x1, 0.0, 1.0)))
    y_min, y_max = sorted((np.clip(y0, 0.0, 1.0), np.clip(y1, 0.0, 1.0)))
    return (x_min, y_min, x_max, y_max)


def estimate_lung_mask(display_img_rgb):
    """
    Rough lung-field mask so the Grad-CAM overlay never colors outside the
    lungs. There's no trained lung-segmentation model in this pipeline, so
    this is a classical CV heuristic: Otsu threshold to isolate darker,
    air-filled tissue from the body, intersected with a fixed anatomical ROI
    (an ellipse over where lungs sit in a centered frontal CXR). The ROI
    intersection matters because the dark-tissue test alone can't tell arms
    from lungs -- they're frequently one connected dark blob through the
    armpit, so filtering by connected-component centroid lets the whole
    blob (shoulders/arms included) through. It's an approximation, not
    ground truth.
    """
    gray = cv2.cvtColor(display_img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Anatomical prior: lungs sit centrally, below the clavicles/shoulders and
    # above the abdomen, well inset from the left/right edges where the arms
    # are. An ellipse (vs. a rectangle) also tapers the corners in, which is
    # exactly where the shoulder/neck area otherwise leaks through.
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        roi_mask,
        center=(w // 2, int(h * 0.52)),
        axes=(int(w * 0.36), int(h * 0.38)),
        angle=0, startAngle=0, endAngle=360,
        color=255, thickness=-1,
    )

    # Body mask: exclude the near-black background/border first.
    _, body_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    body_pixels = gray[body_mask > 0]
    if body_pixels.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Otsu split within the body: darker (air-filled) tissue = lung candidate.
    otsu_thresh, _ = cv2.threshold(body_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = ((gray < otsu_thresh) & (body_mask > 0)).astype(np.uint8) * 255

    # Close small holes (ribs/vessels crossing the lung field), then drop noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    lung_mask = cv2.bitwise_and(dark_mask, roi_mask)

    # Drop any leftover small/noisy fragments inside the ROI.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lung_mask, connectivity=8)
    min_area = 0.01 * h * w
    cleaned = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, num_labels):  # label 0 is the background component
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    return cleaned


class ExplainabilityEngine:
    """
    Generates Grad-CAM heatmaps and bounding box overlays for an already
    loaded component3 diagnostic model.
    """

    def __init__(self, model):
        self.model = model

    def preprocess_for_explainability(self, image_bytes: bytes):
        """Applies the Bilateral -> CLAHE -> Unsharp sequence directly from bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # Preprocessing Pipeline
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        img_tensor = np.expand_dims(final_rgb.astype('float32') / 255.0, axis=0)
        return final_rgb, img_tensor

    def find_target_layer(self):
        """
        Targets `concatenate_1` (112x112x128) -- the finer-resolution of the two
        parallel branches that feed the shared classification/bbox feature vector
        (the other is conv2d_5 at 14x14, 8x coarser). Picked by direct graph
        inspection of the loaded model rather than a generic shape heuristic,
        because list-order-based layer selection (last 4D layer in model.layers
        matching a spatial-size predicate) is unreliable on this multi-branch
        architecture -- model.layers reflects definition order, not depth, so it
        was landing on the coarse branch and producing a blocky, unfocused heatmap.
        """
        target_layer_name = 'concatenate_1'
        self.model.get_layer(target_layer_name)  # raises if the model is ever swapped and this layer no longer exists
        print(f"Targeting layer for Grad-CAM: {target_layer_name}")
        return target_layer_name

    def generate_dual_explanation(self, image_bytes: bytes) -> str:
        """
        Generates the heatmap and bounding box directly from image bytes.
        Returns the superimposed image as a Base64 encoded string for the API.
        """
        # 1. Preprocess
        display_img, img_tensor = self.preprocess_for_explainability(image_bytes)

        # 2. Find Target Layer
        target_layer_name = self.find_target_layer()

        # 3. Build Gradient Model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(target_layer_name).output, self.model.output]
        )

        # 4. Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            class_preds = predictions['class_output'][0]
            predicted_class_idx = tf.argmax(class_preds)
            loss = class_preds[predicted_class_idx]

        # 5. Compute Gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # 6. Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 7. Apply ReLU (Drop negative gradients)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        # Cast off float16 (mixed-precision conv layers) since cv2.resize has no float16 kernel
        heatmap = tf.cast(heatmap, tf.float32).numpy()

        # 8. Overlay Heatmap onto Enhanced Image -- full-range JET heatmap
        # (blue = low activation -> red/yellow = high), gated to the estimated
        # lung fields so nothing outside the lungs picks up color. The mask
        # edge is feathered (Gaussian blur) so the color blends smoothly into
        # the surrounding anatomy instead of a hard-edged cutout.
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        lung_mask = estimate_lung_mask(display_img)
        soft_mask = cv2.GaussianBlur(lung_mask, (15, 15), 0).astype(np.float32) / 255.0
        alpha = (soft_mask * 0.45)[..., np.newaxis]
        overlaid = (display_img.astype(np.float32) * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)

        # 9. Locate the strongest localized hotspot(s) within the lung fields --
        # the top ~5% of activation, intersected with the lung mask, grouped
        # into connected regions and ranked by peak intensity. These drive the
        # labeled boxes below. This replaces the old bbox_output-driven box,
        # which was known to be corner-pinned garbage (see clip_and_order_bbox
        # callers elsewhere) -- these boxes come from the real computed
        # heatmap instead. The heatmap is smoothed first so scattered
        # high-frequency speckle (this branch's raw activation is noisy pixel
        # to pixel) collapses into a few coherent focal peaks instead of one
        # blob covering most of the lung once morphology closes the gaps.
        smoothed = cv2.GaussianBlur(heatmap_resized, (9, 9), 0)
        lung_activation = smoothed[lung_mask > 0]
        hotspot_thresh = np.percentile(lung_activation, 95) if lung_activation.size else 1.0
        hotspot_mask = ((smoothed >= hotspot_thresh) & (lung_mask > 0)).astype(np.uint8) * 255
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_OPEN, close_kernel, iterations=1)

        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(hotspot_mask, connectivity=8)
        min_hotspot_area = 0.003 * 224 * 224
        max_hotspot_area = 0.06 * 224 * 224  # skip components too large/diffuse to be a focal lesion
        regions = []
        for label in range(1, num_labels):  # label 0 is the background component
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_hotspot_area or area > max_hotspot_area:
                continue
            x, y, rw, rh = (stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP],
                             stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT])
            peak_val = float(heatmap_resized[cc_labels == label].max())
            regions.append((peak_val, x, y, rw, rh))
        regions.sort(key=lambda r: r[0], reverse=True)
        regions = regions[:2]

        # 10. Compose the final canvas: upscale for presentation quality, then
        # add a dark margin for a title bar, labeled leader lines (kept off
        # the X-ray itself), and a legend -- all driven by the real regions
        # computed above, nothing fabricated.
        SCALE = 1.5
        core = cv2.resize(overlaid, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        core_size = core.shape[0]
        PAD_TOP, PAD_BOTTOM, PAD_SIDE = 46, 20, 120
        canvas_h, canvas_w = core_size + PAD_TOP + PAD_BOTTOM, core_size + 2 * PAD_SIDE
        canvas = np.full((canvas_h, canvas_w, 3), (18, 22, 30), dtype=np.uint8)
        canvas[PAD_TOP:PAD_TOP + core_size, PAD_SIDE:PAD_SIDE + core_size] = core

        cv2.putText(canvas, "AI-Assisted TB Detection - Grad-CAM Visualization", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 220, 230), 1, cv2.LINE_AA)

        box_color = (80, 220, 255)
        for i, (peak_val, x, y, rw, rh) in enumerate(regions, start=1):
            bx0, by0 = int(x * SCALE) + PAD_SIDE, int(y * SCALE) + PAD_TOP
            bx1, by1 = bx0 + int(rw * SCALE), by0 + int(rh * SCALE)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), box_color, 1, cv2.LINE_AA)

            label = f"TB Lesion {i:02d}"
            anchor_y = (by0 + by1) // 2
            label_y = PAD_TOP + 24 + (i - 1) * 24
            # Alternate label sides so two boxes never collide in the same margin.
            anchor_x, label_x = (bx1, canvas_w - PAD_SIDE + 8) if i == 1 else (bx0, 8)
            cv2.line(canvas, (anchor_x, anchor_y), (label_x, label_y), box_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (label_x, label_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 1, cv2.LINE_AA)

        legend_x, legend_y = canvas_w - PAD_SIDE + 8, canvas_h - 130
        cv2.putText(canvas, "TB Attention", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (215, 220, 230), 1, cv2.LINE_AA)
        legend_items = [
            ((40, 90, 210), "Low"),
            ((60, 190, 110), "Mild"),
            ((225, 215, 60), "Moderate"),
            ((240, 150, 40), "High"),
            ((225, 50, 40), "Very High"),
        ]
        for idx, (color, text) in enumerate(legend_items):
            y = legend_y + 16 + idx * 18
            cv2.rectangle(canvas, (legend_x, y - 9), (legend_x + 14, y + 3), color, -1)
            cv2.putText(canvas, text, (legend_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (200, 205, 215), 1, cv2.LINE_AA)

        # 11. Convert directly to Base64 String (Bypassing Hard Drive)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.png', canvas_bgr)
        if not success:
            raise ValueError("Failed to encode Grad-CAM image to PNG buffer.")
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        return heatmap_base64
