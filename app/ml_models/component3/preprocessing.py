import cv2
import numpy as np

IMG_SIZE = (224, 224)


def apply_clinical_preprocessing(image_bytes):
    """
    Applies the training-time clinical preprocessing sequence: decode -> resize
    -> grayscale -> bilateral denoise -> CLAHE -> unsharp mask -> RGB.

    This mirrors ExplainabilityEngine.preprocess_for_explainability in gradcam.py
    exactly, so the tensor fed to model.predict() matches the tensor Grad-CAM
    builds its heatmap from. Returns the (1, 224, 224, 3) normalized tensor
    ready for model.predict().
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    img_tensor = np.expand_dims(final_rgb.astype('float32') / 255.0, axis=0)

    return img_tensor
