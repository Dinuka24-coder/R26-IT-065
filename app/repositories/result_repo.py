from bson import ObjectId
from app.database import get_database

async def save_result(collection_name: str, result: dict) -> str:
    db = get_database()
    res = await db[collection_name].insert_one(result)
    return str(res.inserted_id)

async def get_result_by_id(collection_name: str, result_id: str) -> dict:
    db = get_database()
    result = await db[collection_name].find_one({"_id": ObjectId(result_id)})
    if result:
        result["_id"] = str(result["_id"])
    return result

async def get_results_by_patient(collection_name: str, patient_id: str) -> list:
    db = get_database()
    cursor = db[collection_name].find({"patient_id": patient_id})
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results