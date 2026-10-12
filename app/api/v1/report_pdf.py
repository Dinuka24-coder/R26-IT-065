from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pymongo.errors import PyMongoError
from app.core.dependencies import get_current_user
from app.database import get_database
from app.services import report_service as rs

router = APIRouter(prefix="/reports/pdf", tags=["PDF Reports"])


def _pdf_response(pdf: bytes, filename: str, inline: bool) -> Response:
    disposition = "inline" if inline else "attachment"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


async def _filtered_rows(db, user, start_date, end_date, disease, status, patient_id, doctor_id, search):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(400, "start_date must be before end_date")
    try:
        rows = await rs.gather_results(db, user, start_date, end_date, disease, status, patient_id, doctor_id)
        await rs.attach_names(db, rows)
    except PyMongoError as e:
        print(f"[reports] database error: {type(e).__name__}: {e}")
        raise HTTPException(503, "Database is not responding. Check your internet connection and try again.")
    return rs.search_rows(rows, search)


@router.get("/list")
async def list_results(
    start_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    disease: Optional[str] = "all",
    status: Optional[Literal["all", "Positive", "Normal"]] = "all",
    patient_id: Optional[str] = None,
    doctor_id: Optional[str] = Query(None, description="Admin only"),
    search: Optional[str] = Query(None, description="Patient, doctor, disease or result text"),
    user=Depends(get_current_user),
):
    """Results for the Reports page (JSON). Doctors only get their own results."""
    db = get_database()
    rows = await _filtered_rows(db, user, start_date, end_date, disease, status, patient_id, doctor_id, search)
    for r in rows:
        r["created_at"] = r["created_at"].isoformat() if r["created_at"] else None
    return {"role": user.get("role"), "count": len(rows), "rows": rows}


@router.get("/summary")
async def summary_report(
    start_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="YYYY-MM-DD"),
    disease: Optional[str] = "all",
    status: Optional[Literal["all", "Positive", "Normal"]] = "all",
    patient_id: Optional[str] = None,
    doctor_id: Optional[str] = Query(None, description="Admin only"),
    search: Optional[str] = None,
    inline: bool = Query(False, description="true = open in browser, false = download"),
    user=Depends(get_current_user),
):
    """Summary PDF. With doctor_id / disease / patient_id it becomes a doctor, disease or patient report."""
    db = get_database()
    rows = await _filtered_rows(db, user, start_date, end_date, disease, status, patient_id, doctor_id, search)

    doctor_label = None
    if doctor_id and user.get("role") == "admin":
        doctor_label = await rs.get_user_name(db, doctor_id) or doctor_id

    patient_label = None
    if patient_id:
        p = await rs.get_patient(db, patient_id) or {}
        code = p.get("patient_id") or patient_id
        name = p.get("full_name") or p.get("name") or ""
        patient_label = f"{code} - {name}" if name else code

    one_disease = disease and disease.lower() != "all"
    if patient_label:
        title, slug = f"Patient Report - {patient_label}", f"patient_{patient_label.split(' - ')[0]}"
    elif doctor_label:
        title, slug = f"Doctor Report - {doctor_label}", "doctor"
    elif one_disease:
        title, slug = f"Disease Report - {disease}", f"disease_{disease.replace(' ', '')}"
    else:
        title, slug = "Diagnostic Summary Report", "summary"

    pdf = rs.build_summary_pdf(
        rows,
        {"title": title, "start_date": start_date, "end_date": end_date, "disease": disease, "status": status,
         "patient": patient_label, "doctor": doctor_label, "search": search},
        user.get("full_name") or user.get("email", ""),
    )
    return _pdf_response(pdf, f"PulmoAI_{slug}_{date.today().isoformat()}.pdf", inline)


@router.get("/result/{component}/{result_id}")
async def result_report(
    component: Literal["pneumothorax", "pneumonia", "tuberculosis", "lungcancer"],
    result_id: str,
    inline: bool = Query(False, description="true = open in browser, false = download"),
    user=Depends(get_current_user),
):
    """PDF for one prediction (one patient, one scan)."""
    db = get_database()
    result = await rs.get_result(db, component, result_id)
    if not result:
        raise HTTPException(404, "Result not found")

    if user.get("role") != "admin" and str(result.get("doctor_id", "")) != rs.user_id(user):
        raise HTTPException(403, "You can only download reports for your own patients")

    patient = await rs.get_patient(db, result.get("patient_id"))
    doctor_name = await rs.get_user_name(db, str(result.get("doctor_id", "")))
    pdf = rs.build_result_pdf(result, component, patient, doctor_name,
                              user.get("full_name") or user.get("email", ""))
    return _pdf_response(pdf, f"PulmoAI_{component}_{result.get('patient_id', 'patient')}_{result_id[-6:]}.pdf", inline)


@router.get("/screening")
async def screening_report(
    pneumothorax: Optional[str] = Query(None, description="Pneumothorax result ID"),
    pneumonia: Optional[str] = Query(None, description="Pneumonia result ID"),
    tuberculosis: Optional[str] = Query(None, description="Tuberculosis result ID"),
    lungcancer: Optional[str] = Query(None, description="Lung cancer result ID"),
    inline: bool = Query(False, description="true = open in browser, false = download"),
    user=Depends(get_current_user),
):
    """One PDF for a Full Screening (same patient, several disease results)."""
    ids = {"pneumothorax": pneumothorax, "pneumonia": pneumonia,
           "tuberculosis": tuberculosis, "lungcancer": lungcancer}
    ids = {k: v for k, v in ids.items() if v}
    if not ids:
        raise HTTPException(400, "Give at least one result ID")

    db = get_database()
    items = []
    for component, rid in ids.items():
        result = await rs.get_result(db, component, rid)
        if not result:
            raise HTTPException(404, f"{component} result not found")
        if user.get("role") != "admin" and str(result.get("doctor_id", "")) != rs.user_id(user):
            raise HTTPException(403, "You can only download reports for your own patients")
        items.append((component, result))

    if len({str(r.get("patient_id")) for _, r in items}) > 1:
        raise HTTPException(400, "All results must belong to the same patient")

    first = items[0][1]
    patient = await rs.get_patient(db, first.get("patient_id"))
    doctor_name = await rs.get_user_name(db, str(first.get("doctor_id", "")))
    pdf = rs.build_screening_pdf(items, patient, doctor_name, user.get("full_name") or user.get("email", ""))
    code = (patient or {}).get("patient_id") or first.get("patient_id", "patient")
    return _pdf_response(pdf, f"PulmoAI_full_screening_{code}_{date.today().isoformat()}.pdf", inline)