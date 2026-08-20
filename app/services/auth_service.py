from fastapi import HTTPException
from app.core.security import verify_password, create_access_token
from app.repositories.user_repo import get_user_by_email


async def login_user(email: str, password: str):
    user = await get_user_by_email(email)
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
        "sub":  str(user["_id"]),
        "role": user["role"],
    })

    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":                   str(user["_id"]),
            "full_name":            user["full_name"],
            "email":                user["email"],
            "role":                 user["role"],
            "contact_number":       user.get("contact_number"),
            "registered_number":    user.get("registered_number"),
            "must_change_password": user.get("must_change_password", False),
        }
    }