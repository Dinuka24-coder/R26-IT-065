from app.database import get_database
from bson import ObjectId


def _users():
    return get_database()["users"]


async def create_user(user_doc: dict):
    result = await _users().insert_one(user_doc)
    return str(result.inserted_id)


async def get_user_by_email(email: str):
    return await _users().find_one({"email": email})


async def get_user_by_id(user_id: str):
    try:
        return await _users().find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


async def get_all_doctors():
    cursor = _users().find({"role": "doctor"})
    return [doc async for doc in cursor]


async def update_user(user_id: str, update_data: dict):
    await _users().update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )


async def delete_user(user_id: str):
    result = await _users().delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count > 0