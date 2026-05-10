from tensorflow.keras.models import load_model
import os

#WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "pneumothorax_model.keras")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "pneumothorax_model_final.keras")

model = None

def get_model():
    global model
    if model is None:
        model = load_model(WEIGHTS_PATH)
        print("✅ Pneumothorax model loaded")
        print("Model layers:")
        for i, layer in enumerate(model.layers):
            print(f"  [{i}] {layer.name}")
    return model