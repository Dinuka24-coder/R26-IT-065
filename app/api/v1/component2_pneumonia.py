from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import cv2
import numpy as np
import logging

from app.ml_models.component2.inference import run_pneumonia_inference, InvalidXRayError
from app.services.comp2_service import save_pneumonia_prediction

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def pneumonia_status():
    return {"message": "Component 2 - Pneumonia endpoint is working"}

@router.post("/predict")
async def predict_pneumonia(
    patient_id: str = Form(...), # NEW: Ask the user for the Patient ID
    include_explanation_image: bool = Form(False),
    file: UploadFile = File(...)
):
    # 1. Read and decode the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")

    try:
        # 2. Run the AI Math
        diagnosis, confidence, severity, heatmap_base64, heatmap_sev = run_pneumonia_inference(img)

        # 3. Save to MongoDB using the Service
        db_record_id = await save_pneumonia_prediction(
            patient_id=patient_id,
            filename=file.filename,
            diagnosis=diagnosis,
            confidence=confidence,
            severity=severity,
            heatmap_base64=heatmap_base64
        )

        # 4. Return the result to the screen (including quantitative attention metrics)
        return {
            "database_record_id": db_record_id,
            "patient_id": patient_id,
            "filename": file.filename,
            "diagnosis": diagnosis,
            "confidence": f"{confidence:.2f}%",
            "affected_area_percent": heatmap_sev["affected_area_percent"],
            "mean_intensity": heatmap_sev["mean_intensity"],
            "explanation_image": heatmap_base64 if include_explanation_image else None
        }

    except InvalidXRayError as e:
        logger.warning(f"OOD Shield validation failed for file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pneumonia inference or database save failed")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")