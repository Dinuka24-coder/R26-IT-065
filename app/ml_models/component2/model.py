import tensorflow as tf
import os

# Get the absolute path to where the weights are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "best_mobilenetv2_pneumonia.keras")

def load_pneumonia_model():
    """Loads the MobileNetV2 model for Pneumonia detection."""
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at: {WEIGHTS_PATH}")
    
    print(f"Loading Component 2 Brain from {WEIGHTS_PATH}...")
    model = tf.keras.models.load_model(WEIGHTS_PATH)
    return model

# Create a single instance to be shared across the application
pneumonia_model = load_pneumonia_model()