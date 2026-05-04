from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import cv2
import numpy as np
import logging

from app.ml_models.component2.inference import run_pneumonia_inference
from app.services.comp2_service import save_pneumonia_prediction

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def pneumonia_status():
    return {"message": "Component 2 - Pneumonia endpoint is working"}

@router.post("/predict")
async def predict_pneumonia(
    patient_id: str = Form(...), # NEW: Ask the user for the Patient ID
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
        diagnosis, confidence, severity, heatmap_base64 = run_pneumonia_inference(img)

        # 3. Save to MongoDB using the Service
        db_record_id = await save_pneumonia_prediction(
            patient_id=patient_id,
            filename=file.filename,
            diagnosis=diagnosis,
            confidence=confidence,
            severity=severity,
            heatmap_base64=heatmap_base64
        )

        # 4. Return the result to the screen (now including the Database ID!)
        return {
            "database_record_id": db_record_id,
            "patient_id": patient_id,
            "filename": file.filename,
            "diagnosis": diagnosis,
            "confidence": f"{confidence:.2f}%",
            "severity": severity,
            "explanation_image": "Base64 String Omitted for Console Speed" # Kept short for testing
        }

    except Exception as e:
        logger.exception("Pneumonia inference or database save failed")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")