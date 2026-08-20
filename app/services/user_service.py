from fastapi import HTTPException
from datetime import datetime
from app.core.security import hash_password, verify_password
from app.repositories.user_repo import (
    create_user, get_user_by_email, get_all_doctors,
    update_user, delete_user, get_user_by_id
)



async def create_doctor(data: dict):
    existing = await get_user_by_email(data["email"])
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    doctor_doc = {
        "full_name":         data["full_name"],
        "email":             data["email"],
        "contact_number":    data["contact_number"],
        "registered_number": data["registered_number"],
        "password":          hash_password(data["password"]),
        "role":              "doctor",
        "created_at":        datetime.utcnow().isoformat(),
    }
    doctor_id = await create_user(doctor_doc)
    return {"id": doctor_id, "message": "Doctor created successfully"}


async def list_doctors():
    doctors = await get_all_doctors()
    return [
        {
            "id":                str(d["_id"]),
            "full_name":         d["full_name"],
            "email":             d["email"],
            "contact_number":    d.get("contact_number"),
            "registered_number": d.get("registered_number"),
            "created_at":        d.get("created_at"),
        }
        for d in doctors
    ]


async def update_doctor_profile(doctor_id: str, data: dict, current_user: dict):
    # A doctor can only edit their OWN profile (admin can edit any)
    if current_user["role"] == "doctor" and str(current_user["_id"]) != doctor_id:
        raise HTTPException(status_code=403, detail="You can only edit your own profile")

    update_data = {k: v for k, v in data.items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    await update_user(doctor_id, update_data)
    return {"message": "Profile updated successfully"}


async def remove_doctor(doctor_id: str):
    user = await get_user_by_id(doctor_id)
    if not user:
        raise HTTPException(status_code=404, detail="Doctor not found")
    if user["role"] == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin")

    await delete_user(doctor_id)
    return {"message": "Doctor deleted successfully"}

async def reset_doctor_password(doctor_id: str, new_password: str):
    """Admin resets a doctor's password. Forces change on next login."""
    user = await get_user_by_id(doctor_id)
    if not user:
        raise HTTPException(status_code=404, detail="Doctor not found")
    if user.get("role") == "admin":
        raise HTTPException(status_code=400, detail="Cannot reset admin password here")

    await update_user(doctor_id, {
        "password": hash_password(new_password),
        "must_change_password": True,
    })
    return {"message": "Password reset successfully. The doctor must change it on next login."}


async def change_own_password(current_user: dict, current_password: str, new_password: str):
    """Any logged-in user changes their own password."""
    if not verify_password(current_password, current_user["password"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if current_password == new_password:
        raise HTTPException(status_code=400, detail="New password must be different")

    await update_user(str(current_user["_id"]), {
        "password": hash_password(new_password),
        "must_change_password": False,
    })
    return {"message": "Password changed successfully"}