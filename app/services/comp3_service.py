from fastapi import UploadFile, HTTPException
from app.ml_models.component3.inference import predict_tuberculosis
from app.models.component3_schema import TBPredictionResponse

async def process_tb_scan(file: UploadFile) -> TBPredictionResponse:
    try:
        # 1. Read the raw bytes from the uploaded HTTP file
        image_bytes = await file.read()
        
        # 2. Pass bytes to your isolated AI Engine
        raw_result = predict_tuberculosis(image_bytes)
        
        # 3. (Optional Future Step) Save result to MongoDB here
        # db_record = {"model": "GhostNet_MobileViT", "result": raw_result}
        # await save_to_db(db_record)
        
        # 4. Return the validated Pydantic model
        return TBPredictionResponse(**raw_result)
        
    except Exception as e:
        # Catch any preprocessing or mathematical errors gracefully
        raise HTTPException(status_code=500, detail=f"AI Processing Pipeline Error: {str(e)}")