import io
import numpy as np
from PIL import Image
from app.ml_models.component3.model import tb_model_instance

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

def preprocess_tuberculosis(image_bytes: bytes) -> np.ndarray:
    """
    Component 3 specific preprocessing: 
    RGB conversion, 224x224 resize, and 0.0 - 1.0 normalization.
    """
    # 1. Load image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. Resize to match Colab training input exactly
    img = img.resize(IMG_SIZE)
    
    # 3. Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 4. Add batch dimension -> Shape becomes (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

def predict_tuberculosis(image_bytes: bytes) -> dict:
    """
    Executes the GhostNet + MobileViT prediction pipeline.
    """
    # 1. Preprocess the raw bytes
    img_array = preprocess_tuberculosis(image_bytes)
    
    # 2. Load model and predict
    model = tb_model_instance.load_model()
    # verbose=0 keeps the FastAPI console clean from progress bars
    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    
    # 3. Format clinical output
    if raw_score >= THRESHOLD:
        label = "Tuberculosis Detected"
        confidence = round(raw_score * 100, 2)
    else:
        label = "Normal"
        confidence = round((1 - raw_score) * 100, 2)
        
    return {
        "prediction": label,
        "confidence": confidence,
        "raw_score": round(raw_score, 4),
        # A flag for the frontend to alert the doctor if the AI is unsure
        "requires_clinical_review": 40.0 <= confidence <= 60.0 
    }