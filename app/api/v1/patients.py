from fastapi import APIRouter, Depends
from app.models.patient_model import PatientCreate, PatientUpdate, ClinicalNoteCreate
from app.core.dependencies import require_doctor_or_admin
from app.services.patient_service import (
    register_patient, find_patients, list_all_patients,
    get_patient, edit_patient, remove_patient, add_note
)

router = APIRouter(prefix="/patients", tags=["Patient Management"])


@router.post("")
async def create(data: PatientCreate, user=Depends(require_doctor_or_admin)):
    return await register_patient(data.dict(), user)


@router.get("/search")
async def search(q: str, user=Depends(require_doctor_or_admin)):
    return await find_patients(q)


@router.get("")
async def list_patients(user=Depends(require_doctor_or_admin)):
    return await list_all_patients()


@router.get("/{patient_id}")
async def get_one(patient_id: str, user=Depends(require_doctor_or_admin)):
    return await get_patient(patient_id)


@router.put("/{patient_id}")
async def update(patient_id: str, data: PatientUpdate,
                 user=Depends(require_doctor_or_admin)):
    return await edit_patient(patient_id, data.dict())


@router.delete("/{patient_id}")
async def delete(patient_id: str, user=Depends(require_doctor_or_admin)):
    return await remove_patient(patient_id)


@router.post("/{patient_id}/notes")
async def add_clinical_note(patient_id: str, data: ClinicalNoteCreate,
                            user=Depends(require_doctor_or_admin)):
    return await add_note(patient_id, data.note, user)