from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.comp1_service import run_prediction

router = APIRouter()

@router.post("/predict")
async def predict_pneumothorax(
    patient_id: str        = Form(...),
    file:       UploadFile = File(...)
):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    result      = await run_prediction(patient_id, image_bytes)
    return result