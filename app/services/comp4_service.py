from app.ml_models.component4.inference import predict
from app.repositories.result_repo import save_result
from datetime import datetime, timezone


async def run_prediction(patient_id: str, image_bytes: bytes) -> dict:
    # Run model prediction
    result = predict(image_bytes)

    # Build clean response (ONLY what you want)
    final_result = {
        "patient_id": patient_id,
        "component": "CT-Based-Lung-cancer-Classification",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Save to MongoDB
    saved_id = await save_result("lung_cancer_results", final_result)

    # Add result id for tracking
    final_result["result_id"] = str(saved_id)

    return final_result