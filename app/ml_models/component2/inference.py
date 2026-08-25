import cv2
import numpy as np
import base64
import logging
import tensorflow as tf
from .model import get_pneumonia_model, get_autoencoder, get_ood_shield_data, get_tb_shield_data
from .gradcam import make_gradcam_heatmap
from .severity import calculate_pneumonia_severity, calculate_heatmap_severity

logger = logging.getLogger(__name__)

class InvalidXRayError(ValueError):
    """Exception raised when an image is not recognized as an authentic chest X-ray."""
    pass


def _apply_clahe(img_rgb):
    """Applies CLAHE in LAB color space, matching the training preprocessing exactly.
    
    Args:
        img_rgb: 224x224 RGB uint8 image.
        
    Returns:
        CLAHE-enhanced RGB uint8 image (224x224).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced


def run_pneumonia_inference(original_img):
    """Processes an image and returns diagnosis, confidence, severity, and XAI heatmap.
    
    Pipeline:
        1. Preprocess: resize 224x224 + CLAHE (LAB color space)
        2. OOD gatekeeper: autoencoder embedding → Euclidean distance to pneumonia centroid
        3. TB shield: reject if within TB cluster and closer to TB than pneumonia centroid
        4. Classification: MobileNetV2 with preprocess_input ([-1, 1] scaling)
        5. Grad-CAM + severity metrics
    """
    # --- Step 1: Preprocess — resize + CLAHE ---
    img_resized = cv2.resize(original_img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_clahe = _apply_clahe(img_rgb)
    
    # --- Step 2: OOD Gatekeeper (autoencoder) ---
    # Autoencoder expects [0, 1] normalized input
    img_ae = np.expand_dims(img_clahe, axis=0).astype('float32') / 255.0
    
    _, encoder = get_autoencoder()
    embedding = encoder.predict(img_ae, verbose=0)
    feat_vector = embedding.flatten()
    
    lung_centroid, ood_threshold = get_ood_shield_data()
    distance = np.linalg.norm(feat_vector - lung_centroid)
    
    logger.info(f"OOD Shield - Distance: {distance:.4f}, Threshold: {ood_threshold:.4f}")
    
    if distance > ood_threshold:
        raise InvalidXRayError("Input a valid pneumonia xray")

    # --- Step 3: TB Shield (separate from OOD gatekeeper) ---
    tb_centroid, tb_rejection_radius = get_tb_shield_data()
    tb_distance = np.linalg.norm(feat_vector - tb_centroid)

    logger.info(
        f"TB Shield - TB distance: {tb_distance:.4f}, "
        f"Pneumonia distance: {distance:.4f}, "
        f"TB radius: {tb_rejection_radius:.4f}"
    )

    if tb_distance <= tb_rejection_radius and distance > tb_distance:
        raise InvalidXRayError("Input a valid pneumonia xray")
    
    # --- Step 4: Classification (MobileNetV2) ---
    # preprocess_input expects uint8 or float, scales to [-1, 1]
    img_classifier = np.expand_dims(img_clahe.copy(), axis=0).astype('float32')
    img_classifier = tf.keras.applications.mobilenet_v2.preprocess_input(img_classifier)
    
    pneumonia_model = get_pneumonia_model()
    prediction = pneumonia_model.predict(img_classifier, verbose=0)
    pneumonia_chance = float(prediction[0][0] * 100)
    
    # --- Step 5: Threshold + Diagnosis ---
    severity = calculate_pneumonia_severity(pneumonia_chance)
    
    heatmap_base64 = None
    if pneumonia_chance >= 47:
        diagnosis = "PNEUMONIA DETECTED"
        
        # --- Step 6: Grad-CAM Heatmap ---
        heatmap = make_gradcam_heatmap(img_classifier, pneumonia_model)
        
        # Superimpose heatmap onto the original image (alpha=0.4, COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Encode to base64
        success, buffer = cv2.imencode('.jpg', superimposed_img)
        if not success:
            raise RuntimeError("Failed to encode heatmap image to JPEG.")
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # --- Step 7: Heatmap-based Severity Metrics ---
        heatmap_severity = calculate_heatmap_severity(heatmap)
        logger.info(
            f"Heatmap severity - Affected area: {heatmap_severity['affected_area_percent']:.2f}%, "
            f"Mean intensity: {heatmap_severity['mean_intensity']:.4f}"
        )
    else:
        diagnosis = "NORMAL"
        heatmap_severity = {
            "affected_area_percent": 0.0,
            "mean_intensity": 0.0
        }
        
    return diagnosis, pneumonia_chance, severity, heatmap_base64, heatmap_severity
