
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import connect_db, close_db, get_database
from app.config import settings
from app.api.v1.router import router
from fastapi.staticfiles import StaticFiles



app = FastAPI(
    title="Pulmonary CDSS API",
    description="Unified Clinical Decision Support System for Pulmonary Diseases",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Register routes
app.include_router(router, prefix=settings.API_PREFIX)

# DB lifecycle
@app.on_event("startup")
async def startup():
    await connect_db()

@app.on_event("shutdown")
async def shutdown():
    await close_db()

@app.get("/")
async def root():
    return {"message": "Pulmonary CDSS API is running"}

@app.get("/test-db")
async def test_db():
    db = get_database()
    collections = await db.list_collection_names()
    return {
        "status": "✅ Connected",
        "database": settings.MONGO_DB_NAME,
        "collections": collections
    }