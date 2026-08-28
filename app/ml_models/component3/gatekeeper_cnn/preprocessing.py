import cv2
import numpy as np

from app.ml_models.component3.gatekeeper_cnn.config import IMG_SIZE
from app.utils.image_utils import safe_decode_image


def preprocess_for_gatekeeper(image_bytes: bytes) -> np.ndarray:
    """Decode -> resize -> grayscale-stacked-to-3-channel -> raw float32.

    Deliberately NOT component3/preprocessing.py's clinical pipeline
    (bilateral/CLAHE/unsharp): that pipeline is tuned for X-ray contrast, and
    applying it ahead of a "is this even a CXR" decision is semantically
    backwards -- it also couples the gatekeeper to unrelated future changes
    in the clinical preprocessing. No /255 normalization here: MobileNetV3's
    built-in Rescaling layer (include_preprocessing=True in model.py) expects
    raw [0,255] floats and handles normalization internally.

    3-channel stacking of a grayscale image (rather than decoding true color)
    matches the existing precedent in component1/inference.py.

    Returns a (1, 224, 224, 3) float32 batch ready for model.predict().
    """
    gray = safe_decode_image(image_bytes, mode=cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(gray, IMG_SIZE)
    stacked = np.stack([resized, resized, resized], axis=-1).astype("float32")
    return np.expand_dims(stacked, axis=0)
