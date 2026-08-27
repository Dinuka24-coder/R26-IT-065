import logging
import os
import numpy as np
import tensorflow as tf

from app.config import settings
from app.ml_models.component3.preprocessing import apply_clinical_preprocessing
from app.ml_models.component3.gatekeeper import EuclideanCXRGatekeeper
from app.ml_models.component3.gatekeeper_cnn.inference import CXRGatekeeperCNN
from app.ml_models.component3.gatekeeper_cnn.gradcam import GatekeeperExplainer
from app.ml_models.component3.gatekeeper_openai.inference import OpenAICXRGatekeeper
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

    def __init__(self, model_path=None, gatekeeper_cnn_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(current_dir, 'weights', 'multi_task_diagnostic_model.keras')
        centroid_path = os.path.join(current_dir, 'weights', 'master_cxr_centroid.npy')

        # Stage 1 is an always-on cascade: heuristic (cheap, real-data-
        # calibrated against all 12,278 TBX11K images) -> OpenAI vision model
        # (primary CXR/quality judgment, generalizes far better to real non-
        # CXR content than anything trained only on this repo's synthetic
        # negatives) -> local CNN (automatic fallback if OpenAI errors, times
        # out, or no API key is configured). The heuristic stays in front of
        # both -- not just as a load-failure fallback -- because it's free
        # and catches degenerate uploads (flat/noise/blank) before spending
        # an API call or a model forward-pass on them.
        self.heuristic_gatekeeper = EuclideanCXRGatekeeper(centroid_path=centroid_path)

        try:
            self.cnn_gatekeeper = CXRGatekeeperCNN(model_path=gatekeeper_cnn_path)
            self.gatekeeper_explainer = GatekeeperExplainer(model=self.cnn_gatekeeper.model)
        except Exception:
            logging.exception("Gatekeeper CNN failed to load; unavailable as a fallback")
            self.cnn_gatekeeper = None
            self.gatekeeper_explainer = None

        # Fail-open: no key configured means every request just skips straight
        # to the CNN+heuristic path below, exactly like local dev already
        # works today. Never crashes app startup over a missing key.
        if settings.OPENAI_API_KEY:
            self.openai_gatekeeper = OpenAICXRGatekeeper(
                api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL
            )
        else:
            logging.info("OPENAI_API_KEY not set; gatekeeper will use CNN+heuristic only")
            self.openai_gatekeeper = None

        self.diagnostic_model = tf.keras.models.load_model(
            model_path, custom_objects={'masked_bbox_loss': masked_bbox_loss}
        )

        self.explain_engine = ExplainabilityEngine(model=self.diagnostic_model)
        self.classes = CLASSES

    async def _run_gatekeeper_cascade(self, image_bytes):
        """Runs Stage 1a (heuristic) -> Stage 1b (OpenAI, if configured) ->
        Stage 1c (local CNN, only reached if OpenAI is unconfigured or its
        call fails). Returns (gate: dict, backend: str). gate always has the
        shared {is_cxr, cxr_confidence, quality_score, accepted, reason}
        shape. This method is async because the OpenAI stage is a real
        network call; the heuristic and CNN stages are still plain sync
        calls underneath."""
        gate = self.heuristic_gatekeeper.inspect_image_detailed(image_bytes)
        if not gate["accepted"]:
            return gate, "heuristic"

        if self.openai_gatekeeper is not None:
            try:
                gate = await self.openai_gatekeeper.inspect_image_detailed(image_bytes)
                return gate, ("heuristic+openai" if gate["accepted"] else "openai")
            except Exception:
                logging.exception("OpenAI gatekeeper call failed; falling back to CNN+heuristic")
                # falls through to the CNN fallback below

        if self.cnn_gatekeeper is None:
            return gate, "heuristic_only_cnn_unavailable"

        gate = self.cnn_gatekeeper.inspect_image_detailed(image_bytes)
        backend = "heuristic+cnn_fallback" if gate["accepted"] else "cnn_fallback"
        return gate, backend

    async def inspect_gatekeeper_only(self, image_bytes):
        """Runs just Stage 1 (the gatekeeper cascade), skipping clinical
        preprocessing and TB inference entirely. Used by the standalone
        /gatekeeper/predict QA endpoint -- never writes to the database and
        never implies a TB diagnosis either way."""
        gate, backend = await self._run_gatekeeper_cascade(image_bytes)

        # Grad-CAM only exists for the local CNN -- OpenAI's hosted model
        # exposes no gradients, so a heatmap is only ever possible when the
        # CNN actually ran (i.e. the "_fallback" backends).
        gatekeeper_heatmap_base64 = None
        if gate["accepted"] and self.gatekeeper_explainer is not None and "cnn_fallback" in backend:
            gatekeeper_heatmap_base64 = self.gatekeeper_explainer.generate(image_bytes)

        return {
            "accepted": gate["accepted"],
            "is_cxr": gate["is_cxr"],
            "cxr_confidence": gate["cxr_confidence"],
            "quality_score": gate["quality_score"],
            "reason": gate["reason"],
            "gatekeeper_backend": backend,
            "gatekeeper_heatmap_base64": gatekeeper_heatmap_base64,
        }

    async def process_scan(self, image_bytes, patient_id):
        """
        Executes the end-to-end clinical flow: Gatekeeper -> Preprocess ->
        Predict -> Conditional Grad-CAM. Returns a plain dict payload.
        """
        # STAGE 1: Gatekeeper cascade (heuristic -> OpenAI -> CNN fallback)
        gate, backend = await self._run_gatekeeper_cascade(image_bytes)
        if not gate["accepted"]:
            return {
                "status": "rejected",
                "patient_id": patient_id,
                "message": gate["reason"],
                "is_cxr": gate["is_cxr"],
                "cxr_confidence": gate["cxr_confidence"],
                "quality_score": gate["quality_score"],
                "gatekeeper_backend": backend,
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
            "is_cxr": gate["is_cxr"],
            "cxr_confidence": gate["cxr_confidence"],
            "quality_score": gate["quality_score"],
            "gatekeeper_backend": backend,
        }

        # Only run the heavier Grad-CAM math if TB was actually predicted
        if diagnosis == 'Tuberculosis':
            heatmap_base64 = self.explain_engine.generate_dual_explanation(image_bytes)

            response_payload["bounding_box"] = list(clip_and_order_bbox(bbox_coords))
            response_payload["heatmap_base64"] = heatmap_base64
            response_payload["clinical_note"] = "Tuberculosis detected. Visual localization heatmap generated."

        return response_payload
