import numpy as np
import cv2
from app.ml_models.component1.model import get_model

IMG_SIZE  = (224, 224)
THRESHOLD = 0.4

def preprocess(image_bytes: bytes) -> np.ndarray:
    # Convert bytes to image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img   = cv2.resize(img, IMG_SIZE)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Normalize
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=-1)
    return np.expand_dims(img, axis=0)

def predict(image_bytes: bytes) -> dict:
    model     = get_model()
    img       = preprocess(image_bytes)
    raw_score = float(model.predict(img, verbose=0)[0][0])

    if raw_score >= THRESHOLD:
        label      = "Pneumothorax Detected"
        confidence = round(raw_score * 100, 2)
    else:
        label      = "Normal"
        confidence = round((1 - raw_score) * 100, 2)

    return {
        "prediction": label,
        "confidence": confidence,
        "raw_score":  round(raw_score, 4)
    }