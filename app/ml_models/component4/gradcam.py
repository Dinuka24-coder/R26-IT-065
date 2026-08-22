import os
import uuid
import cv2
import numpy as np
import tensorflow as tf

from app.ml_models.component4.model import get_model
from app.ml_models.component4.inference import preprocess


HEATMAP_DIR = "static/gradcam/component4"
os.makedirs(HEATMAP_DIR, exist_ok=True)


def generate_gradcam(
    image_bytes: bytes,
    model=None,
    layer_name: str = "texture_conv_4",
    return_heatmap_only: bool = False,
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
    """
    # model=None preserves EXACT existing behavior for every current
    # caller (PNG/JPG path) - only get_model() (the original PNG/JPG
    # model) is used unless a caller explicitly passes a different
    # model (the DICOM model, from comp4_service.py's DICOM-specific
    # functions). Same texture_conv_4 target layer works for both,
    # since both share the same architecture.
    if model is None:
        model = get_model()

    img_array = preprocess(image_bytes)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        predicted_index = tf.argmax(predictions[0])
        class_output = predictions[:, predicted_index]

    grads = tape.gradient(class_output, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    nparr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original is None:
        raise ValueError("Invalid image file for Grad-CAM")

    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    filename = f"{uuid.uuid4()}_gradcam.png"
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