from fastapi import APIRouter, Depends
from app.models.user_model import DoctorCreate, DoctorUpdate, PasswordReset, PasswordChange
from app.core.dependencies import require_admin, get_current_user
from app.services.user_service import (
    create_doctor, list_doctors, update_doctor_profile, remove_doctor, reset_doctor_password, change_own_password
)

router = APIRouter(prefix="/users", tags=["User Management"])


# ── Admin creates a doctor ─────────────────────────────────────
@router.post("/doctors")
async def add_doctor(data: DoctorCreate, admin=Depends(require_admin)):
    return await create_doctor(data.dict())


# ── Admin lists all doctors ────────────────────────────────────
@router.get("/doctors")
async def get_doctors(admin=Depends(require_admin)):
    return await list_doctors()


# ── Doctor edits own profile (or admin edits any) ──────────────
@router.put("/doctors/{doctor_id}")
async def edit_doctor(doctor_id: str, data: DoctorUpdate,
                      current_user=Depends(get_current_user)):
    return await update_doctor_profile(doctor_id, data.dict(), current_user)


# ── Admin deletes a doctor ─────────────────────────────────────
@router.delete("/doctors/{doctor_id}")
async def delete_doctor(doctor_id: str, admin=Depends(require_admin)):
    return await remove_doctor(doctor_id)


# ── Get own profile ────────────────────────────────────────────
@router.get("/me")
async def get_me(current_user=Depends(get_current_user)):
    return {
        "id":                str(current_user["_id"]),
        "full_name":         current_user["full_name"],
        "email":             current_user["email"],
        "role":              current_user["role"],
        "contact_number":    current_user.get("contact_number"),
        "registered_number": current_user.get("registered_number"),
    }

# ── Admin resets a doctor's password ───────────────────────────
@router.post("/doctors/{doctor_id}/reset-password")
async def reset_password(doctor_id: str, data: PasswordReset,
                         admin=Depends(require_admin)):
    return await reset_doctor_password(doctor_id, data.new_password)


# ── Logged-in user changes their own password ──────────────────
@router.post("/me/change-password")
async def change_password(data: PasswordChange,
                          current_user=Depends(get_current_user)):
    return await change_own_password(
        current_user, data.current_password, data.new_password
    )