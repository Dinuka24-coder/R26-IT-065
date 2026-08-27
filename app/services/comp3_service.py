from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException
from app.ml_models.component3.controller import DiagnosticController
from app.models.component3_schema import TBPredictionResponse, GatekeeperResponse
from app.repositories.result_repo import save_result
from app.utils.image_utils import validate_upload_bytes

# Loaded once at import time (mirrors the singleton pattern used by other
# components) since it holds the ~13MB multi-task model in memory.
_controller = DiagnosticController()


async def process_tb_scan(patient_id: str, file: UploadFile) -> TBPredictionResponse:
    try:
        image_bytes = await file.read()
        try:
            validate_upload_bytes(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = await _controller.process_scan(image_bytes, patient_id)

        if result["status"] == "rejected":
            # A gatekeeper rejection is a normal, expected verdict -- not an
            # error -- so it comes back as a 200 with status="rejected" and the
            # full structured detail (is_cxr / cxr_confidence / quality_score /
            # gatekeeper_backend / message) the frontend needs to explain *why*
            # the scan was not analyzed. No DB record is written for a rejected
            # upload (we return before save_result below). This is distinct from
            # status="error" further down, which is a real pipeline failure and
            # stays a 500.
            return TBPredictionResponse(
                patient_id=patient_id,
                filename=file.filename,
                status="rejected",
                message=result["message"],
                is_cxr=result.get("is_cxr"),
                cxr_confidence=result.get("cxr_confidence"),
                quality_score=result.get("quality_score"),
                gatekeeper_backend=result.get("gatekeeper_backend"),
            )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        db_record = {
            "patient_id": patient_id,
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


async def run_gatekeeper_only(file: UploadFile) -> GatekeeperResponse:
    """QA/manual-testing utility: runs just the gatekeeper cascade (heuristic
    -> OpenAI -> CNN fallback), skipping clinical preprocessing and TB
    inference entirely. Never writes to the database and never implies a TB
    diagnosis."""
    try:
        image_bytes = await file.read()
        try:
            validate_upload_bytes(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = await _controller.inspect_gatekeeper_only(image_bytes)
        result["filename"] = file.filename

        return GatekeeperResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gatekeeper Processing Error: {str(e)}")
