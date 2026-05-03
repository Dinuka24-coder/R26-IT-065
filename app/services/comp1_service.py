from app.ml_models.component1.inference import predict
from app.ml_models.component1.urgency import classify_urgency
from app.repositories.result_repo import save_result
from datetime import datetime

async def run_prediction(patient_id: str, image_bytes: bytes) -> dict:
    result  = predict(image_bytes)
    urgency = classify_urgency(result["confidence"], result["prediction"])

    final_result = {
        "patient_id": patient_id,
        "component":  "pneumothorax",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "raw_score":  result["raw_score"],
        "urgency":    urgency,
        "created_at": datetime.utcnow().isoformat()
    }

    saved_id = await save_result("pneumothorax_results", final_result)

    # ── Remove _id added by MongoDB before returning ──────────
    final_result.pop("_id", None)
    final_result["result_id"] = str(saved_id)

    return final_result