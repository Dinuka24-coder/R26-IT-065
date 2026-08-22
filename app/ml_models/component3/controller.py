import os
import numpy as np
import tensorflow as tf

from app.ml_models.component3.preprocessing import apply_clinical_preprocessing
from app.ml_models.component3.gatekeeper import EuclideanCXRGatekeeper
from app.ml_models.component3.gradcam import ExplainabilityEngine, clip_and_order_bbox

CLASSES = ['Healthy', 'Non-TB', 'Tuberculosis']


@tf.keras.utils.register_keras_serializable(package='Custom')
def masked_bbox_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    box_sums = tf.reduce_sum(y_true, axis=-1)
    mask = tf.cast(box_sums > 0.0, tf.float32)
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return tf.reduce_mean(mse * mask)


class DiagnosticController:
    """
    Orchestrates the full component3 (Tuberculosis) inference pipeline:
    gatekeeper -> clinical preprocessing -> multi-task model -> conditional
    Grad-CAM + bounding box explanation.
    """

    def __init__(self, model_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(current_dir, 'weights', 'multi_task_diagnostic_model.keras')
        centroid_path = os.path.join(current_dir, 'weights', 'master_cxr_centroid.npy')

        self.gatekeeper = EuclideanCXRGatekeeper(centroid_path=centroid_path)

        self.diagnostic_model = tf.keras.models.load_model(
            model_path, custom_objects={'masked_bbox_loss': masked_bbox_loss}
        )

        self.explain_engine = ExplainabilityEngine(model=self.diagnostic_model)
        self.classes = CLASSES

    def process_scan(self, image_bytes, patient_id):
        """
        Executes the end-to-end clinical flow: Gatekeeper -> Preprocess ->
        Predict -> Conditional Grad-CAM. Returns a plain dict payload.
        """
        # STAGE 1: Gatekeeper
        is_valid_cxr, gate_msg = self.gatekeeper.inspect_image(image_bytes)
        if not is_valid_cxr:
            return {
                "status": "rejected",
                "patient_id": patient_id,
                "message": gate_msg,
            }

        # STAGE 2: Preprocessing
        try:
            input_tensor = apply_clinical_preprocessing(image_bytes)
        except ValueError as e:
            return {"status": "error", "patient_id": patient_id, "message": f"Preprocessing failed: {str(e)}"}

        # STAGE 3: Inference (Keras 3 multi-task dict output)
        predictions = self.diagnostic_model.predict(input_tensor, verbose=0)
        class_probs = predictions['class_output'][0]
        bbox_coords = predictions['bbox_output'][0]

        predicted_idx = int(np.argmax(class_probs))
        diagnosis = self.classes[predicted_idx]
        confidence = float(class_probs[predicted_idx]) * 100.0

        # STAGE 4: Conditional explainability + output
        response_payload = {
            "status": "success",
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "confidence_score": round(confidence, 2),
            "message": "Valid CXR. Analysis complete.",
        }

        # Only run the heavier Grad-CAM math if TB was actually predicted
        if diagnosis == 'Tuberculosis':
            heatmap_base64 = self.explain_engine.generate_dual_explanation(image_bytes)

            response_payload["bounding_box"] = list(clip_and_order_bbox(bbox_coords))
            response_payload["heatmap_base64"] = heatmap_base64
            response_payload["clinical_note"] = "Tuberculosis detected. Visual localization heatmap generated."

        return response_payload
