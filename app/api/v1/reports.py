import asyncio
from fastapi import APIRouter, Depends
from app.core.dependencies import require_doctor_or_admin
from app.database import get_database
from app.utils.collection_cache import collection_exists

router = APIRouter(prefix="/reports", tags=["Reports"])

COMPONENT_COLLECTIONS = {
    "pneumothorax_results": ("Pneumothorax", "X-ray"),
    "pneumonia_results":    ("Pneumonia",    "X-ray"),
    "tuberculosis_results": ("Tuberculosis", "X-ray"),
    "lung_cancer_results":  ("Lung Cancer",  "CT Scan"),
}

QUERY_TIMEOUT = 10.0
MAX_DOCS      = 1000

POSITIVE_MARKERS = ("detected", "positive", "malignant", "abnormal")
NEGATIVE_MARKERS = ("normal", "no ", "negative", "clear")
CANCER_MARKERS   = ("carcinoma", "adenocarcinoma", "nodule")


def _is_positive(record) -> bool:
    """Normalize the varying prediction wording used across components."""
    text = str(record.get("prediction") or record.get("diagnosis") or "").lower().strip()
    if not text:
        return False
    if any(m in text for m in NEGATIVE_MARKERS):
        return False
    if any(m in text for m in POSITIVE_MARKERS):
        return True
    # Component 4 returns raw cancer subtype names
    if any(c in text for c in CANCER_MARKERS):
        return True
    return str(record.get("status", "")).lower() == "positive"


def _get_created_at(r):
    """Component 3 stores only `timestamp`; the others store `created_at`."""
    if r.get("created_at"):
        return r["created_at"]
    ts = r.get("timestamp")
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts) if ts else None


def _disease_label(record, positive: bool, default_label: str, collection: str) -> str:
    """Show the specific cancer subtype rather than a generic label."""
    if not positive:
        return "Normal"
    if collection == "lung_cancer_results":
        raw = str(record.get("prediction") or "").strip()
        if raw:
            return raw.replace(".", " ").title()
    return default_label


async def _safe_to_list(cursor, label: str, limit: int = MAX_DOCS):
    """Run a cursor with a hard timeout; return [] on failure."""
    try:
        return await asyncio.wait_for(cursor.to_list(length=limit), timeout=QUERY_TIMEOUT)
    except Exception as e:
        print(f"⚠️ Query failed on {label}: {type(e).__name__}")
        return []


@router.get("/history")
async def history_report(patient_id: str = None, doctor_id: str = None,
                         user=Depends(require_doctor_or_admin)):
    """
    Diagnostic history across all component collections.
    Optional filters: patient_id, doctor_id
    """
    db = get_database()

    # ── Name lookups ───────────────────────────────────────────
    patients = await _safe_to_list(db["patients"].find(), "patients")
    patient_map = {str(p["_id"]): p.get("full_name", "Unknown") for p in patients}

    doctors = await _safe_to_list(
        db["users"].find({"role": {"$in": ["doctor", "admin"]}}), "users", limit=200
    )
    doctor_map = {str(d["_id"]): d.get("full_name", "Unknown") for d in doctors}

    # ── Gather results ─────────────────────────────────────────
    records = []
    for coll, (disease, scan) in COMPONENT_COLLECTIONS.items():
        if not collection_exists(coll):      # instant — no DB round trip
            continue

        query = {}
        if patient_id:
            query["patient_id"] = patient_id
        if doctor_id:
            query["doctor_id"] = doctor_id

        docs = await _safe_to_list(db[coll].find(query), coll)

        for r in docs:
            positive = _is_positive(r)
            records.append({
                "patient_id":   r.get("patient_id"),
                "patient_name": patient_map.get(str(r.get("patient_id")), "—"),
                "doctor_id":    r.get("doctor_id"),
                "doctor_name":  doctor_map.get(str(r.get("doctor_id")), "—"),
                "date":         _get_created_at(r),
                "disease":      _disease_label(r, positive, disease, coll),
                "scan_type":    scan,
                "confidence":   float(r.get("confidence") or 0),
                "status":       "Positive" if positive else "Normal",
            })

    records.sort(key=lambda x: x["date"] or "", reverse=True)
    return {"total": len(records), "records": records}