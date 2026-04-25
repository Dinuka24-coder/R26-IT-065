from fastapi import FastAPI
from app.database import connect_db, close_db
from app.config import settings
from app.api.v1.router import router

app = FastAPI(
    title="Pulmonary CDSS API",
    description="Unified Clinical Decision Support System for Pulmonary Diseases",
    version="1.0.0"
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