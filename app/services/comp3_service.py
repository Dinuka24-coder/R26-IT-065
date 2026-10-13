from datetime import datetime, timezone
from typing import Optional

from fastapi import UploadFile, HTTPException
from app.ml_models.component3.controller import DiagnosticController
from app.models.component3_schema import TBPredictionResponse, GatekeeperResponse
from app.repositories.result_repo import save_result
from app.utils.image_utils import validate_upload_bytes

# Loaded once at import time (mirrors the singleton pattern used by other
# components) since it holds the ~13MB multi-task model in memory.
_controller = DiagnosticController()

# Model labels -> text shown in the app and reports (same as Full Screening)
TB_DISPLAY = {
    "Healthy":      "Normal",
    "Non-TB":       "Other Abnormality",
    "Tuberculosis": "Tuberculosis Detected",
}


def tb_urgency(diagnosis: str, confidence: float) -> str:
    """Same urgency rule as Full Screening."""
    if diagnosis != "Tuberculosis":
        return "Low"
    if confidence >= 85:
        return "High"
    if confidence >= 65:
        return "Moderate"
    return "Low"


async def process_tb_scan(patient_id: str, file: UploadFile, user: Optional[dict] = None) -> TBPredictionResponse:
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

        diagnosis = result["diagnosis"]                       # Healthy | Non-TB | Tuberculosis
        confidence = float(result["confidence_score"])
        urgency = tb_urgency(diagnosis, confidence)
        now = datetime.now(timezone.utc)

        # Same shape as the Full Screening TB record, so Reports, Dashboard,
        # History and PDFs treat both the same way.
        db_record = {
            "status":           "success",
            "is_xray":          True,
            "patient_id":       patient_id,
            "doctor_id":        str(user["_id"]) if user else None,
            "component":        "tuberculosis",
            "filename":         file.filename,
            "prediction":       TB_DISPLAY.get(diagnosis, diagnosis),
            "diagnosis":        diagnosis,
            "confidence":       confidence,
            "confidence_score": confidence,          # kept for older code that reads this name
            "severity":         urgency,
            "urgency":          urgency,
            "bounding_box":     result.get("bounding_box"),
            "clinical_note":    result.get("clinical_note"),
            "created_at":       now.isoformat(),
            "timestamp":        now,                 # kept for older code that reads this name
        }

        inserted_id = await save_result(collection_name="tuberculosis_results", result=db_record)
        saved_id = str(getattr(inserted_id, "inserted_id", inserted_id))

        result["database_record_id"] = saved_id
        result["result_id"] = saved_id               # same name as the other components (PDF button)
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