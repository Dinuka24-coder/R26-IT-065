from bson import ObjectId
from app.database import get_database

async def create_patient(patient_data: dict) -> str:
    db = get_database()
    result = await db["patients"].insert_one(patient_data)
    return str(result.inserted_id)

async def get_patient_by_id(patient_id: str) -> dict:
    db = get_database()
    patient = await db["patients"].find_one({"_id": ObjectId(patient_id)})
    if patient:
        patient["_id"] = str(patient["_id"])
    return patient

async def get_all_patients() -> list:
    db = get_database()
    cursor = db["patients"].find()
    patients = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        patients.append(doc)
    return patients