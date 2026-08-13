from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from app.services.auth_service import login_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # OAuth2 form uses "username" field — we treat it as email
    return await login_user(form_data.username, form_data.password)