from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Patient registration ───────────────────────────────────────
class PatientCreate(BaseModel):
    full_name:      str
    age:            int
    gender:         str
    nic:            str
    contact_number: str
    address:        str
    clinical_notes: Optional[str] = ""


# ── Patient update ─────────────────────────────────────────────
class PatientUpdate(BaseModel):
    full_name:      Optional[str] = None
    age:            Optional[int] = None
    gender:         Optional[str] = None
    nic:            Optional[str] = None
    contact_number: Optional[str] = None
    address:        Optional[str] = None


# ── Add a dated clinical note ───────────────────────────────────
class ClinicalNoteCreate(BaseModel):
    note: str