from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def pneumothorax_status():
    return {"message": "Component 1 - Pneumothorax endpoint is working"}