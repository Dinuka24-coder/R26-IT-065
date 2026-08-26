from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException
from app.ml_models.component3.controller import DiagnosticController
from app.models.component3_schema import TBPredictionResponse
from app.repositories.result_repo import save_result
 
# Loaded once at import time (mirrors the singleton pattern used by other
# components) since it holds the ~13MB multi-task model in memory.
_controller = DiagnosticController()
 
 
async def process_tb_scan(patient_id: str, file: UploadFile,
                          current_user: dict = None) -> TBPredictionResponse:

    print(f"Processing TB scan for patient_id: {patient_id}, filename: {file.filename}")

    try:
        image_bytes = await file.read()
 
        result = _controller.process_scan(image_bytes, patient_id)
 
        if result["status"] == "rejected":
            raise HTTPException(status_code=422, detail=result["message"])
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
 
        db_record = {
            "patient_id": patient_id,
            "doctor_id": str(current_user["_id"]) if current_user else None,
            "component": "tuberculosis",
            "filename": file.filename,
            "diagnosis": result["diagnosis"],
            "confidence_score": result["confidence_score"],
            "bounding_box": result.get("bounding_box"),
            "clinical_note": result.get("clinical_note"),
            "timestamp": datetime.now(timezone.utc),
        }
 
        inserted_id = await save_result(collection_name="tuberculosis_results", result=db_record)
 
        result["database_record_id"] = str(inserted_id)
        result["filename"] = file.filename
 
        return TBPredictionResponse(**result)
 
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Pipeline Error: {str(e)}")