import numpy as np
import cv2

from app.ml_models.component4.model import get_model

#IMG_SIZE = (256, 256)
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

def preprocess(image_bytes: bytes) -> np.ndarray:

    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    return img

def predict(image_bytes: bytes) -> dict:

    model = get_model()

    img = preprocess(image_bytes)

    prediction = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(np.max(prediction))

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "raw_scores": {
            CLASS_NAMES[i]: round(float(score), 4)
            for i, score in enumerate(prediction)
        }
    }