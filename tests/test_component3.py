# test_component3.py
import numpy as np
from PIL import Image
import io
from app.ml_models.component3.inference import predict_tuberculosis

def create_dummy_image_bytes() -> bytes:
    """Creates a fake image in memory to simulate an API file upload."""
    # Create a random noise image
    random_pixels = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    img = Image.fromarray(random_pixels)
    
    # Save it to a bytes buffer
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def test_local_pipeline():
    print("--- TESTING COMPONENT 3 INFERENCE PIPELINE ---")
    
    # Simulate receiving bytes from FastAPI
    print("Simulating image upload...")
    image_bytes = create_dummy_image_bytes()
    
    # Run the full inference
    print("Running preprocessing and prediction...")
    try:
        result = predict_tuberculosis(image_bytes)
        print("[SUCCESS] Pipeline successful! Result:")
        print(result)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    test_local_pipeline()
