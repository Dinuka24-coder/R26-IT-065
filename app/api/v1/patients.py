from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_patients():
    return {"message": "Patients endpoint is working"}