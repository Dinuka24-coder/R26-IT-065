from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# ── Doctor creation (by admin) ─────────────────────────────────
class DoctorCreate(BaseModel):
    full_name:         str
    email:             EmailStr
    contact_number:    str
    registered_number: str
    password:          str


# ── Doctor profile update (by doctor themselves) ───────────────
class DoctorUpdate(BaseModel):
    full_name:         Optional[str] = None
    email:             Optional[EmailStr] = None
    contact_number:    Optional[str] = None
    registered_number: Optional[str] = None


# ── Login ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

# ── Admin resets a doctor's password ───────────────────────────
class PasswordReset(BaseModel):
    new_password: str = Field(..., min_length=6, max_length=72)


# ── User changes their own password ────────────────────────────
class PasswordChange(BaseModel):
    current_password: str
    new_password:     str = Field(..., min_length=6, max_length=72)


# ── Response ───────────────────────────────────────────────────
class UserResponse(BaseModel):
    id:                str
    full_name:         str
    email:             str
    role:              str
    contact_number:    Optional[str] = None
    registered_number: Optional[str] = None
    created_at:        Optional[str] = None