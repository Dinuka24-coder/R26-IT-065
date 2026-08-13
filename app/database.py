from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client: AsyncIOMotorClient = None
db = None

async def connect_db():
    global client, db
    client = AsyncIOMotorClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=20000,   # fail fast instead of 30s
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
        maxPoolSize=50,  # reuse connections
        retryWrites=True
    )
    db = client[settings.MONGO_DB_NAME]
    print(f"✅ Connected to MongoDB: {settings.MONGO_DB_NAME}")

async def close_db():
    global client
    if client:
        client.close()
        print("🔌 MongoDB connection closed")

def get_database():
    return client[settings.MONGO_DB_NAME]