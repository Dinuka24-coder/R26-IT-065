from tensorflow.keras.models import load_model
import os

# Configurable path - to replace with a better DICOM model later, just
# swap the .keras file at this path (or point this constant at a new
# filename). Inference code (dicom_inference.py) never needs to change
# as long as the replacement model keeps the same contract: input
# (224,224,3), output (4), same class order as DICOM_CLASS_NAMES in
# dicom_inference.py.
DICOM_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "weights",
    "dicom_dual_path_lung_model.keras"
)

_dicom_model = None


def get_dicom_model():
    global _dicom_model

    if _dicom_model is None:
        _dicom_model = load_model(DICOM_WEIGHTS_PATH)
        print("✅ DICOM Lung Cancer Model Loaded")

    return _dicom_model