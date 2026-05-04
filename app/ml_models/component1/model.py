from tensorflow.keras.models import load_model
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "pneumothorax_model.keras")

model = None

def get_model():
    global model
    if model is None:
        model = load_model(WEIGHTS_PATH)
        print("✅ Pneumothorax model loaded")
    return model