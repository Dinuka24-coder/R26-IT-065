from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException
from app.ml_models.component3.inference import predict_tuberculosis
from app.models.component3_schema import TBPredictionResponse
from app.repositories.result_repo import save_result

async def process_tb_scan(patient_id: str, file: UploadFile) -> TBPredictionResponse:
    try:
        # 1. Read the raw bytes from the uploaded HTTP file
        image_bytes = await file.read()
        
        # 2. Pass bytes to your isolated AI Engine
        raw_result = predict_tuberculosis(image_bytes)
        
        # 3. Save result to MongoDB
        db_record = {
            "patient_id": patient_id,
            "component": "tuberculosis",
            "filename": file.filename,
            "prediction": raw_result["prediction"],
            "confidence": raw_result["confidence"],
            "raw_score": raw_result["raw_score"],
            "requires_clinical_review": raw_result["requires_clinical_review"],
            "timestamp": datetime.now(timezone.utc)
        }
        
        inserted_id = await save_result(collection_name="tuberculosis_results", result=db_record)
        
        # 4. Add DB info to the result
        raw_result["database_record_id"] = str(inserted_id)
        raw_result["patient_id"] = patient_id
        raw_result["filename"] = file.filename
        
        # 5. Return the validated Pydantic model
        return TBPredictionResponse(**raw_result)
        
    except Exception as e:
        # Catch any preprocessing or mathematical errors gracefully
        raise HTTPException(status_code=500, detail=f"AI Processing Pipeline Error: {str(e)}")