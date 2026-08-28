import base64

import cv2
import numpy as np
import tensorflow as tf

from app.ml_models.component3.gatekeeper_cnn.config import IMG_SIZE
from app.ml_models.component3.gatekeeper_cnn.model import GRADCAM_TARGET_LAYER
from app.ml_models.component3.gatekeeper_cnn.preprocessing import preprocess_for_gatekeeper

CAPTION = "Gatekeeper Attention (Modality/Quality Check) - NOT TB Evidence"


class GatekeeperExplainer:
    """Grad-CAM for the gatekeeper's cxr_head only -- shows what the model
    looked at to decide "is this a chest X-ray", not TB evidence. Quality
    issues (blur, exposure, noise) are diffuse/global rather than spatially
    localized, so a heatmap adds little value there; this deliberately only
    explains the modality decision.

    Kept strictly separate from component3/gradcam.py's TB ExplainabilityEngine:
    different field name wherever this is returned (gatekeeper_heatmap_base64,
    never heatmap_base64), a caption baked directly into the image, and this
    must never be attached to the same response payload as a TB diagnosis
    heatmap -- a caller should never be able to confuse the two.
    """

    def __init__(self, model):
        self.model = model
        self.grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(GRADCAM_TARGET_LAYER).input, model.output["cxr_head"]],
        )

    def generate(self, image_bytes: bytes) -> str:
        input_tensor = preprocess_for_gatekeeper(image_bytes)

        with tf.GradientTape() as tape:
            conv_outputs, cxr_pred = self.grad_model(input_tensor)
            loss = cxr_pred[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = tf.cast(heatmap, tf.float32).numpy()

        display_img = input_tensor[0].astype("uint8")

        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        overlaid = (display_img.astype(np.float32) * 0.6 + heatmap_colored * 0.4).astype(np.uint8)

        SCALE = 1.5
        core = cv2.resize(overlaid, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        core_size = core.shape[0]
        PAD_TOP, PAD_BOTTOM, PAD_SIDE = 40, 16, 16
        canvas_h, canvas_w = core_size + PAD_TOP + PAD_BOTTOM, core_size + 2 * PAD_SIDE
        canvas = np.full((canvas_h, canvas_w, 3), (18, 22, 30), dtype=np.uint8)
        canvas[PAD_TOP:PAD_TOP + core_size, PAD_SIDE:PAD_SIDE + core_size] = core

        cv2.putText(canvas, CAPTION, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (215, 220, 230), 1, cv2.LINE_AA)

        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", canvas_bgr)
        if not success:
            raise ValueError("Failed to encode gatekeeper Grad-CAM image to PNG buffer.")
        return base64.b64encode(buffer).decode("utf-8")
