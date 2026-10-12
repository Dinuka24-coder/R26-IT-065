"""
gradcam_core.py - SINGLE authoritative Grad-CAM computation.

Extracted from the previously-duplicated math in gradcam.py's
generate_gradcam(). Every caller that needs a raw Grad-CAM heatmap
uses this function - no other file computes GradientTape gradients
for Grad-CAM independently.
"""

import numpy as np
import tensorflow as tf

from app.ml_models.component4.inference import preprocess


def compute_raw_heatmap(image_bytes: bytes, model, layer_name: str):
    """The one, single Grad-CAM computation used by every caller.
    Returns (raw_heatmap [H,W] float array pre-resize, predicted_index,
    full predictions array)."""
    img_array = preprocess(image_bytes)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
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

    return heatmap.numpy(), int(predicted_index.numpy()), predictions.numpy()[0]