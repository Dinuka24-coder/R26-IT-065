from app.ml_models.component1.boundary_gradcam import generate_boundary_aware_gradcam
from app.ml_models.component1.urgency import classify_urgency
from app.ml_models.component1.xray_validator import is_xray
from app.repositories.result_repo import save_result
from datetime import datetime


async def run_prediction(patient_id: str, image_bytes: bytes, current_user: dict = None):
    validation = is_xray(image_bytes)
    if not validation["is_xray"]:
        return {"status": "rejected", "error": "Uploaded image does not appear to be a chest X-ray.", "is_xray": False}

    result  = generate_boundary_aware_gradcam(image_bytes)
    urgency = classify_urgency(result["confidence"], result["prediction"])

    final_result = {
        "status":              "success",
        "is_xray":             True,
        "patient_id":          patient_id,
        "doctor_id":           str(current_user["_id"]) if current_user else None,  # ← NEW
        "component":           "pneumothorax",
        "prediction":          result["prediction"],
        "confidence":          result["confidence"],
        "raw_score":           result["raw_score"],
        "affected_lung_pct":   result["affected_lung_pct"],
        "boundary_length_pct": result["boundary_length_pct"],
        "pleural_separation":  result["pleural_separation"],
        "segmented_area_pct":  result["segmented_area_pct"],
        "urgency":             urgency,
        "created_at":          datetime.utcnow().isoformat(),
    }

    saved_id = await save_result("pneumothorax_results", final_result.copy())
    final_result.pop("_id", None)
    final_result["result_id"]     = str(saved_id)
    final_result["heatmap_base64"] = result["heatmap_base64"]

    return final_result