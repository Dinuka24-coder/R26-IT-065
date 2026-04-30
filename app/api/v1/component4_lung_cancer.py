from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def lung_cancer_status():
    return {"message": "Component 4 - Lung Cancer endpoint is working"}