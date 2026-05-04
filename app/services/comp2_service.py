from datetime import datetime
from app.repositories.result_repo import save_result

async def save_pneumonia_prediction(patient_id: str, filename: str, diagnosis: str, confidence: float, severity: str, heatmap_base64: str):
    """
    Acts as the middleman: Packages and formats the AI result, 
    then hands it to the repository.
    """
    
    # --- CLEANUP FORMATTING ---
    # 1. Change "PNEUMONIA DETECTED" to "Pneumonia Detected"
    clean_diagnosis = diagnosis.title() 
    
    # 2. Round 99.994239... to just 1 decimal place (e.g., 100.0)
    rounded_confidence = round(confidence, 1) 
    
    # 3. Create a raw score (0.0 to 1.0) just like your teammate did, for future math
    raw_score = confidence / 100.0 

    # --- PACKAGE THE BOX ---
    result_data = {
        "patient_id": patient_id,
        "component": "pneumonia",         # Tagging the component
        "filename": filename,
        "prediction": clean_diagnosis,    # Using the cleaned text
        "confidence": rounded_confidence, # Using the rounded number
        "raw_score": raw_score,           # Adding the raw decimal
        "severity": severity,
        "explanation_image": heatmap_base64,
        "timestamp": datetime.utcnow()    # Kept as a real Date object!
    }

    inserted_id = await save_result(collection_name="pneumonia_results", result=result_data)
    
    return inserted_id