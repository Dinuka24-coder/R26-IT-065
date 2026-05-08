import numpy as np
import cv2
import tensorflow as tf
import base64
from app.ml_models.component1.model import get_model

IMG_SIZE  = (224, 224)
THRESHOLD = 0.3

def preprocess_for_gradcam(image_bytes: bytes) -> tuple:
    nparr = np.frombuffer(image_bytes, np.uint8)

    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_original = cv2.resize(img_original, IMG_SIZE)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img   = cv2.resize(img, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # EfficientNetB0 expects 0-255 NOT normalized
    img   = np.stack([img, img, img], axis=-1).astype(np.float32)

    return np.expand_dims(img, axis=0), img_original


def generate_gradcam(image_bytes: bytes) -> dict:
    model        = get_model()
    img_batch, img_original = preprocess_for_gradcam(image_bytes)
    efficientnet = model.layers[1]  # efficientnetb0

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

        x    = model.layers[2](effnet_output)
        x    = model.layers[3](x)
        x    = model.layers[4](x)
        x    = model.layers[5](x)
        x    = model.layers[6](x)
        x    = model.layers[7](x)
        pred = model.layers[8](x)
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
    heatmap_resized[heatmap_resized < THRESHOLD] = 0

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay         = cv2.addWeighted(
        img_original, 0.6, heatmap_colored, 0.4, 0
    )

    overlay_bgr    = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer      = cv2.imencode(".png", overlay_bgr)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    raw_score  = float(model.predict(img_batch, verbose=0)[0][0])
    label      = "Pneumothorax Detected" if raw_score >= 0.5 else "Normal"
    confidence = round(raw_score * 100, 2) if raw_score >= 0.5 \
                 else round((1 - raw_score) * 100, 2)

    return {
        "prediction":     label,
        "confidence":     confidence,
        "raw_score":      round(raw_score, 4),
        "heatmap_base64": overlay_base64
    }