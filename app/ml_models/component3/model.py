import os
import tensorflow as tf

class TBModelLoader:
    def __init__(self):
        self.model = None
        # Dynamically resolve the absolute path to prevent File Not Found errors
        # regardless of where the FastAPI server is started from.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'weights', 'tb_model.keras')

    def load_model(self):
        # Only load the model if it hasn't been loaded yet
        if self.model is None:
            print(f"Loading Tuberculosis Model from {self.model_path}...")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("[SUCCESS] Component 3 (TB) Model loaded into memory successfully.")
            except Exception as e:
                print(f"[ERROR] CRITICAL ERROR loading model: {e}")
                raise e
        return self.model

# Instantiate the singleton instance to be imported by inference.py
tb_model_instance = TBModelLoader()