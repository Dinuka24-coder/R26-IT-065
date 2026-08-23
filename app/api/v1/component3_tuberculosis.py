from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.comp3_service import process_tb_scan
from app.models.component3_schema import TBPredictionResponse
 
router = APIRouter()
 
@router.post("/predict", response_model=TBPredictionResponse)
async def analyze_tb_scan(
    patient_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a Chest X-Ray image to predict the presence of Tuberculosis.
    """
    print(f"Received request to analyze TB scan for patient_id: {patient_id}, filename: {file.filename}")

    # Quick security check: ensure the user actually uploaded an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
   
    # Pass to the service layer
    result = await process_tb_scan(patient_id, file)
    return result
 