from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.core.dependencies import require_doctor_or_admin
from app.services.comp1_service import run_prediction

router = APIRouter(prefix="/pneumothorax", tags=["Component 1 - Pneumothorax"])


@router.post("/predict")
async def predict_pneumothorax(
    patient_id: str        = Form(...),
    file:       UploadFile = File(...),
    user                   = Depends(require_doctor_or_admin),   # ← auth + doctor identity
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    result      = await run_prediction(patient_id, image_bytes, user)

    if result.get("status") == "rejected":
        raise HTTPException(status_code=422, detail=result["error"])
    return result