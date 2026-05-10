from datetime import datetime, timezone

from app.ml_models.component4.inference import predict
from app.ml_models.component4.gradcam import generate_gradcam
from app.repositories.result_repo import save_result


async def run_prediction(patient_id: str, image_bytes: bytes) -> dict:
    # Run prediction
    result = predict(image_bytes)

    # Generate Grad-CAM heatmap
    heatmap_url = generate_gradcam(image_bytes)

    # Build final response
    final_result = {
        "patient_id": patient_id,
        "component": "CT-Based-Lung-cancer-Classification",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "heatmap_url": heatmap_url,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Save to MongoDB
    saved_id = await save_result(
        "lung_cancer_results",
        final_result
    )

    # Add MongoDB result ID
    final_result["result_id"] = str(saved_id)

    return final_result