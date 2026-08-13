import asyncio
from datetime import datetime
from app.database import connect_db, get_database
from app.core.security import hash_password


async def create_admin():
    await connect_db()
    db = get_database()

    existing = await db["users"].find_one({"email": "admin@pulmocdss.com"})
    if existing:
        print("Admin already exists")
        return

    admin_doc = {
        "full_name":  "System Admin",
        "email":      "admin@pulmocdss.com",
        "password":   hash_password("admin123"),   # CHANGE THIS
        "role":       "admin",
        "created_at": datetime.utcnow().isoformat(),
    }
    await db["users"].insert_one(admin_doc)
    print("✅ Admin created — email: admin@pulmocdss.com, password: admin123")


if __name__ == "__main__":
    asyncio.run(create_admin())