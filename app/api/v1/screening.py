from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.core.dependencies import require_doctor_or_admin
from app.services.screening_service import run_full_screening

router = APIRouter(prefix="/screening", tags=["Full Screening"])


@router.post("/full")
async def full_screening(
    patient_id: str        = Form(...),
    file:       UploadFile = File(...),
    user                   = Depends(require_doctor_or_admin),
):
    """Run one chest X-ray through all available X-ray detection engines."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    result = await run_full_screening(patient_id, image_bytes, user)

    if result.get("status") == "rejected":
        raise HTTPException(status_code=422, detail=result["error"])
    return result