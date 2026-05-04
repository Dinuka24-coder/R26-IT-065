import cv2
import numpy as np
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .model import get_pneumonia_model
from .gradcam import make_gradcam_heatmap
from .severity import calculate_pneumonia_severity

def run_pneumonia_inference(original_img):
    """Processes an image and returns diagnosis, confidence, severity, and XAI heatmap."""
    pneumonia_model = get_pneumonia_model()

    # 1. Preprocess the image — convert BGR (OpenCV default) to RGB for the model
    img = cv2.resize(original_img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = preprocess_input(img_array)
    
    # 2. Get Prediction
    prediction = pneumonia_model.predict(img_array, verbose=0)
    pneumonia_chance = float(prediction[0][0] * 100)
    
    # 3. Calculate Severity
    severity = calculate_pneumonia_severity(pneumonia_chance)
    
    heatmap_base64 = None
    if pneumonia_chance >= 50:
        diagnosis = "PNEUMONIA DETECTED"
        
        # 4. Generate Explainable AI Heatmap
        heatmap = make_gradcam_heatmap(img_array, pneumonia_model)
        
        # Superimpose colors over the original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Convert the superimposed image to Base64 String
        success, buffer = cv2.imencode('.jpg', superimposed_img)
        if not success:
            raise RuntimeError("Failed to encode heatmap image to JPEG.")
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        diagnosis = "NORMAL"
        
    return diagnosis, pneumonia_chance, severity, heatmap_base64
