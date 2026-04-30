from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def tuberculosis_status():
    return {"message": "Component 3 - Tuberculosis endpoint is working"}