import os
import uuid
import cv2
import numpy as np
import tensorflow as tf

from app.ml_models.component4.model import get_model
from app.ml_models.component4.inference import preprocess


HEATMAP_DIR = "static/gradcam/component4"
os.makedirs(HEATMAP_DIR, exist_ok=True)


def generate_gradcam(image_bytes: bytes, layer_name: str = "texture_conv_4") -> str:
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

    return f"/static/gradcam/component4/{filename}"