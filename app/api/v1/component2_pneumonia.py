from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def pneumonia_status():
    return {"message": "Component 2 - Pneumonia endpoint is working"}