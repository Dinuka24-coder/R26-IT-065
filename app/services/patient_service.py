from fastapi import HTTPException
from datetime import datetime
from app.repositories.patient_repo import (
    create_patient, get_patient_by_id, get_patient_by_nic,
    search_patients, get_all_patients, update_patient,
    delete_patient, add_clinical_note
)


def _serialize(patient: dict) -> dict:
    patient["id"] = str(patient["_id"])
    patient.pop("_id", None)
    return patient


async def register_patient(data: dict, current_user: dict):
    existing = await get_patient_by_nic(data["nic"])
    if existing:
        raise HTTPException(status_code=400, detail="Patient with this NIC already exists")

    patient_doc = {
        **data,
        "clinical_notes_history": [],
        "registered_by":          str(current_user["_id"]),
        "created_at":             datetime.utcnow().isoformat(),
    }
    # move initial clinical note into history if provided
    initial_note = patient_doc.pop("clinical_notes", "")
    if initial_note:
        patient_doc["clinical_notes_history"].append({
            "note":       initial_note,
            "date":       datetime.utcnow().isoformat(),
            "doctor_id":  str(current_user["_id"]),
        })

    patient_id = await create_patient(patient_doc)
    return {"id": patient_id, "message": "Patient registered successfully"}


async def find_patients(query: str):
    patients = await search_patients(query)
    return [_serialize(p) for p in patients]


async def list_all_patients():
    patients = await get_all_patients()
    return [_serialize(p) for p in patients]


async def get_patient(patient_id: str):
    patient = await get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _serialize(patient)


async def edit_patient(patient_id: str, data: dict):
    patient = await get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = {k: v for k, v in data.items() if v is not None}
    await update_patient(patient_id, update_data)
    return {"message": "Patient updated successfully"}


async def remove_patient(patient_id: str):
    ok = await delete_patient(patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"message": "Patient deleted successfully"}


async def add_note(patient_id: str, note: str, current_user: dict):
    patient = await get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    note_doc = {
        "note":      note,
        "date":      datetime.utcnow().isoformat(),
        "doctor_id": str(current_user["_id"]),
    }
    await add_clinical_note(patient_id, note_doc)
    return {"message": "Clinical note added", "note": note_doc}