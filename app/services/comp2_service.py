from datetime import datetime, timezone
from typing import Optional
from app.repositories.result_repo import save_result


async def save_pneumonia_prediction(
    patient_id: str,
    filename: str,
    diagnosis: str,
    confidence: float,
    severity: str,
    heatmap_base64: Optional[str] = None,
    affected_area_percent: Optional[float] = None,
    mean_intensity: Optional[float] = None,
    doctor_id: Optional[str] = None,
):
    """
    Packages and formats the AI result matching the unified CDSS schema,
    then persists it to the pneumonia_results MongoDB collection.
    """
    # 1. Standardize diagnosis format (e.g. "Pneumonia Detected", "Normal")
    clean_diagnosis = diagnosis.title()
    if clean_diagnosis.upper() == "PNEUMONIA":
        clean_diagnosis = "Pneumonia Detected"

    # 2. Round confidence to 2 decimal places
    rounded_confidence = round(confidence, 2)

    # 3. Compute raw score (0.0 to 1.0)
    raw_score = round(confidence / 100.0, 4)

    # 4. Package complete document
    result_data = {
        "status": "success",
        "is_xray": True,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "component": "pneumonia",
        "filename": filename,
        "prediction": clean_diagnosis,
        "diagnosis": clean_diagnosis,
        "confidence": rounded_confidence,
        "raw_score": raw_score,
        "severity": severity,
        "affected_area_percent": affected_area_percent,
        "mean_intensity": mean_intensity,
        "explanation_image": heatmap_base64,
        "timestamp": datetime.now(timezone.utc),
        "created_at": datetime.utcnow().isoformat(),
    }

    inserted_id = await save_result(collection_name="pneumonia_results", result=result_data)
    return inserted_id