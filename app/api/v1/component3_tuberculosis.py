from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.comp3_service import process_tb_scan, run_gatekeeper_only
from app.models.component3_schema import TBPredictionResponse, GatekeeperResponse

router = APIRouter()

@router.post("/predict", response_model=TBPredictionResponse)
async def analyze_tb_scan(
    patient_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a Chest X-Ray image to predict the presence of Tuberculosis.
    """
    # Quick security check: ensure the user actually uploaded an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Pass to the service layer
    result = await process_tb_scan(patient_id, file)
    return result


@router.post("/gatekeeper/predict", response_model=GatekeeperResponse)
async def check_gatekeeper_only(file: UploadFile = File(...)):
    """
    QA/manual-testing utility: runs only the CXR Gatekeeper (heuristic ->
    CNN cascade) against an uploaded image, without running TB inference and
    without writing any record to the database. Answers only "is this a
    valid, usable chest X-ray" -- never a TB/clinical judgment.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    result = await run_gatekeeper_only(file)
    return result