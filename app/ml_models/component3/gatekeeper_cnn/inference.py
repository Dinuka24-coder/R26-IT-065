import os

import tensorflow as tf

from app.ml_models.component3.gatekeeper_cnn.config import CXR_THRESHOLD, QUALITY_THRESHOLD, WEIGHTS_FILENAME
from app.ml_models.component3.gatekeeper_cnn.preprocessing import preprocess_for_gatekeeper


class CXRGatekeeperCNN:
    """Loads the trained two-head MobileNetV3Small gatekeeper and exposes the
    same inspect_image_detailed() contract as gatekeeper.py's
    EuclideanCXRGatekeeper, so controller.py can call either polymorphically
    in its heuristic -> CNN cascade."""

    def __init__(self, model_path=None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "weights", WEIGHTS_FILENAME)
            model_path = os.path.normpath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gatekeeper CNN weights not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def inspect_image_detailed(self, image_bytes) -> dict:
        """Returns {is_cxr, cxr_confidence (0-100), quality_score (0-100 or
        None), accepted, reason}. quality_score is forced to None whenever
        is_cxr is False -- the quality_head still emits some sigmoid value for
        non-CXR inputs (loss-masking only affects training, not inference),
        but that number is semantically undefined for e.g. a photo and must
        never be surfaced as if meaningful."""
        input_tensor = preprocess_for_gatekeeper(image_bytes)
        predictions = self.model.predict(input_tensor, verbose=0)

        cxr_prob = float(predictions["cxr_head"][0][0])
        quality_prob = float(predictions["quality_head"][0][0])

        is_cxr = cxr_prob >= CXR_THRESHOLD
        cxr_confidence = round(cxr_prob * 100.0, 2)

        if not is_cxr:
            return {
                "is_cxr": False,
                "cxr_confidence": cxr_confidence,
                "quality_score": None,
                "accepted": False,
                "reason": (
                    f"Rejected: image does not appear to be a chest X-ray "
                    f"(cxr_confidence={cxr_confidence:.2f}%), unsuitable for automated analysis."
                ),
            }

        is_good_quality = quality_prob >= QUALITY_THRESHOLD
        quality_score = round(quality_prob * 100.0, 2)

        if not is_good_quality:
            return {
                "is_cxr": True,
                "cxr_confidence": cxr_confidence,
                "quality_score": quality_score,
                "accepted": False,
                "reason": (
                    f"Rejected: image quality is too low for reliable automated analysis "
                    f"(quality_score={quality_score:.2f}%) -- e.g. blur, poor exposure, rotation, "
                    f"or an incomplete chest region. This does not reflect on any diagnosis."
                ),
            }

        return {
            "is_cxr": True,
            "cxr_confidence": cxr_confidence,
            "quality_score": quality_score,
            "accepted": True,
            "reason": (
                f"Accepted: valid chest X-ray suitable for downstream analysis "
                f"(cxr_confidence={cxr_confidence:.2f}%, quality_score={quality_score:.2f}%)."
            ),
        }
