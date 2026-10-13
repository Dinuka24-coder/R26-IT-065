import asyncio
from bson import ObjectId
from fastapi import APIRouter, Depends
from app.core.dependencies import require_doctor_or_admin
from app.database import get_database
from app.utils.collection_cache import collection_exists

router = APIRouter(prefix="/reports", tags=["Reports"])

# collection -> (disease label, scan type, component key used by the PDF API)
COMPONENT_COLLECTIONS = {
    "pneumothorax_results": ("Pneumothorax", "X-ray",   "pneumothorax"),
    "pneumonia_results":    ("Pneumonia",    "X-ray",   "pneumonia"),
    "tuberculosis_results": ("Tuberculosis", "X-ray",   "tuberculosis"),
    "lung_cancer_results":  ("Lung Cancer",  "CT Scan", "lungcancer"),
}

QUERY_TIMEOUT = 10.0
MAX_DOCS      = 1000

# Only these small fields are downloaded. Heatmaps / images are large and
# made the query time out on MongoDB Atlas.
RESULT_FIELDS  = {"patient_id": 1, "doctor_id": 1, "prediction": 1, "diagnosis": 1,
                  "status": 1, "confidence": 1, "created_at": 1, "timestamp": 1}
PATIENT_FIELDS = {"full_name": 1, "name": 1, "patient_id": 1}
USER_FIELDS    = {"full_name": 1}

POSITIVE_MARKERS = ("detected", "positive", "malignant", "abnormal")
NEGATIVE_MARKERS = ("normal", "no ", "negative", "clear", "other abnormality", "healthy", "non-tb")
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
        v = r["created_at"]
        return v.isoformat() if hasattr(v, "isoformat") else str(v)
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


def _id_filter(value: str):
    """Match a value saved either as a string or as an ObjectId."""
    return {"$in": [value, ObjectId(value)]} if ObjectId.is_valid(value) else value


@router.get("/history")
async def history_report(patient_id: str = None, doctor_id: str = None,
                         user=Depends(require_doctor_or_admin)):
    """
    Diagnostic history across all component collections.
    Optional filters: patient_id, doctor_id
    """
    db = get_database()

    # ── Name lookups (small fields only) ───────────────────────
    patients, doctors = await asyncio.gather(
        _safe_to_list(db["patients"].find({}, PATIENT_FIELDS), "patients", limit=5000),
        _safe_to_list(db["users"].find({"role": {"$in": ["doctor", "admin"]}}, USER_FIELDS), "users", limit=200),
    )

    patient_map = {}
    for p in patients:
        name = p.get("full_name") or p.get("name") or "Unknown"
        patient_map[str(p["_id"])] = name                 # results that store the MongoDB _id
        if p.get("patient_id"):
            patient_map[str(p["patient_id"])] = name      # results that store the code (P001)
    doctor_map = {str(d["_id"]): d.get("full_name", "Unknown") for d in doctors}

    # ── Filters ────────────────────────────────────────────────
    query = {}
    if patient_id:
        # results may store the patient code (P001) or the patient's MongoDB _id
        ids = {patient_id}
        for p in patients:
            if patient_id in (str(p["_id"]), str(p.get("patient_id") or "")):
                ids |= {str(p["_id"]), str(p.get("patient_id") or "")}
        query["patient_id"] = {"$in": [i for i in ids if i]}
    if doctor_id:
        query["doctor_id"] = _id_filter(doctor_id)

    # ── Gather results from all collections at the same time ───
    colls = [c for c in COMPONENT_COLLECTIONS if collection_exists(c)]   # instant — no DB round trip
    all_docs = await asyncio.gather(*(
        _safe_to_list(db[c].find(query, RESULT_FIELDS).sort("_id", -1), c) for c in colls
    ))

    records = []
    for coll, docs in zip(colls, all_docs):
        disease, scan, component = COMPONENT_COLLECTIONS[coll]
        for r in docs:
            positive = _is_positive(r)
            records.append({
                "result_id":    str(r["_id"]),
                "component":    component,
                "patient_id":   str(r.get("patient_id")) if r.get("patient_id") else None,
                "patient_name": patient_map.get(str(r.get("patient_id")), "—"),
                "doctor_id":    str(r.get("doctor_id")) if r.get("doctor_id") else None,
                "doctor_name":  doctor_map.get(str(r.get("doctor_id")), "—"),
                "date":         _get_created_at(r),
                "disease":      _disease_label(r, positive, disease, coll),
                "scan_type":    scan,
                "confidence":   float(r.get("confidence") or 0),
                "status":       "Positive" if positive else "Normal",
            })

    records.sort(key=lambda x: x["date"] or "", reverse=True)
    return {"total": len(records), "records": records}