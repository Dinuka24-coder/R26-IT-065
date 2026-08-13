from app.database import get_database
from bson import ObjectId
from datetime import datetime


def _patients():
    return get_database()["patients"]


async def create_patient(patient_doc: dict):
    result = await _patients().insert_one(patient_doc)
    return str(result.inserted_id)


async def get_patient_by_id(patient_id: str):
    try:
        return await _patients().find_one({"_id": ObjectId(patient_id)})
    except Exception:
        return None


async def get_patient_by_nic(nic: str):
    return await _patients().find_one({"nic": nic})


async def search_patients(query: str):
    cursor = _patients().find({
        "$or": [
            {"full_name": {"$regex": query, "$options": "i"}},
            {"nic":       {"$regex": query, "$options": "i"}},
        ]
    })
    return [doc async for doc in cursor]


async def get_all_patients():
    cursor = _patients().find()
    return [doc async for doc in cursor]


async def update_patient(patient_id: str, update_data: dict):
    await _patients().update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": update_data}
    )


async def delete_patient(patient_id: str):
    result = await _patients().delete_one({"_id": ObjectId(patient_id)})
    return result.deleted_count > 0


async def add_clinical_note(patient_id: str, note_doc: dict):
    await _patients().update_one(
        {"_id": ObjectId(patient_id)},
        {"$push": {"clinical_notes_history": note_doc}}
    )