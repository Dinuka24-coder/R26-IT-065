import tensorflow as tf
import os

# Get the absolute path to where the weights are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "best_mobilenetv2_pneumonia.keras")

_model = None

def get_pneumonia_model():
    """Returns the MobileNetV2 model, loading it lazily on first call."""
    global _model
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights not found at: {WEIGHTS_PATH}")
        print(f"Loading Component 2 model from {WEIGHTS_PATH}...")
        _model = tf.keras.models.load_model(WEIGHTS_PATH)
    return _model