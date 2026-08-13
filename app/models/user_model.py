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


# ── Response ───────────────────────────────────────────────────
class UserResponse(BaseModel):
    id:                str
    full_name:         str
    email:             str
    role:              str
    contact_number:    Optional[str] = None
    registered_number: Optional[str] = None
    created_at:        Optional[str] = None