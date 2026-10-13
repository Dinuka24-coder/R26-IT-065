import asyncio
from fastapi import APIRouter, Depends
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from app.core.dependencies import get_current_user
from app.database import get_database
from app.utils.collection_cache import collection_exists

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

COMPONENT_COLLECTIONS = {
    "pneumothorax_results": ("Pneumothorax", "X-ray"),
    "pneumonia_results":    ("Pneumonia",    "X-ray"),
    "tuberculosis_results": ("Tuberculosis", "X-ray"),
    "lung_cancer_results":   ("Lung Cancer",  "CT Scan"),
}

QUERY_TIMEOUT = 20.0
MAX_DOCS      = 500

# Sri Lanka time (UTC+05:30) — used for "today" and the weekly chart
LOCAL_TZ = timezone(timedelta(hours=5, minutes=30))

# Only these small fields are downloaded. Heatmaps / images are large and
# made the query time out on MongoDB Atlas.
RESULT_FIELDS = {
    "patient_id": 1, "doctor_id": 1, "prediction": 1, "diagnosis": 1,
    "status": 1, "confidence": 1, "created_at": 1, "timestamp": 1,
}

POSITIVE_MARKERS = ("detected", "positive", "malignant", "abnormal")
NEGATIVE_MARKERS = ("normal", "no ", "negative", "clear", "other abnormality", "healthy", "non-tb")

def _is_positive(record) -> bool:
    text = str(record.get("prediction") or record.get("diagnosis") or "").lower().strip()
    if not text:
        return False
    if any(m in text for m in NEGATIVE_MARKERS):
        return False
    if any(m in text for m in POSITIVE_MARKERS):
        return True
    # Component 4 returns raw cancer subtype names
    if any(c in text for c in ("carcinoma", "adenocarcinoma", "nodule")):
        return True
    return str(record.get("status", "")).lower() == "positive"

def _get_created_at(r):
    if r.get("created_at"):
        v = r["created_at"]
        return v.isoformat() if hasattr(v, "isoformat") else str(v)
    ts = r.get("timestamp")
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts) if ts else None

def _local_day(created_at) -> str:
    """UTC time string -> Sri Lanka date 'YYYY-MM-DD' ('' if unknown)."""
    if not created_at:
        return ""
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError:
        return str(created_at)[:10]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ).date().isoformat()

def _today_local():
    return datetime.now(LOCAL_TZ).date()

# ── Shared helpers ─────────────────────────────────────────────
async def _read_collection(db, coll):
    """Newest MAX_DOCS results of one collection (small fields only)."""
    try:
        return await asyncio.wait_for(
            db[coll].find({}, RESULT_FIELDS).sort("_id", -1).to_list(length=MAX_DOCS),
            timeout=QUERY_TIMEOUT
        )
    except Exception as e:
        print(f"⚠️ Skipping {coll}: {type(e).__name__}")
        return []


async def _gather_all_results(db):
    """Collect prediction results from all component collections (in parallel)."""
    colls = [c for c in COMPONENT_COLLECTIONS if collection_exists(c)]   # instant — no DB round trip
    all_docs = await asyncio.gather(*(_read_collection(db, c) for c in colls))

    records = []
    for coll, docs in zip(colls, all_docs):
        disease, scan = COMPONENT_COLLECTIONS[coll]
        for r in docs:
            positive = _is_positive(r)
            records.append({
                "patient_id": str(r.get("patient_id")) if r.get("patient_id") else None,
                "doctor_id":  str(r.get("doctor_id")) if r.get("doctor_id") else None,
                "disease":    disease if positive else "Normal",
                "scan_type":  scan,
                "confidence": r.get("confidence", 0),
                "status":     "Positive" if positive else "Normal",
                "created_at": _get_created_at(r),
            })

    return records


def _build_weekly(results):
    today = _today_local()
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    buckets = {d.isoformat(): {"xray": 0, "ct": 0} for d in days}

    for r in results:
        day = _local_day(r.get("created_at"))
        if day in buckets:
            key = "ct" if r["scan_type"] == "CT Scan" else "xray"
            buckets[day][key] += 1

    return [
        {
            "day":  datetime.fromisoformat(d).strftime("%a"),
            "xray": buckets[d]["xray"],
            "ct":   buckets[d]["ct"],
        }
        for d in buckets
    ]


def _build_distribution(results):
    counts = defaultdict(int)
    for r in results:
        counts[r["disease"]] += 1

    total = sum(counts.values()) or 1
    colors = {
        "Pneumonia": "#3b82f6", "Pneumothorax": "#ef4444",
        "Tuberculosis": "#f59e0b", "Lung Cancer": "#8b5cf6", "Normal": "#22c55e",
    }

    return [
        {"name": k, "value": v, "percent": round(v / total * 100),
         "color": colors.get(k, "#94a3b8")}
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ]


def _build_stats(results, total_patients, total_doctors, role):
    total_scans = len(results)
    positive    = sum(1 for r in results if r["status"] == "Positive")
    today       = _today_local().isoformat()
    today_count = sum(1 for r in results if _local_day(r.get("created_at")) == today)

    return {
        "total_patients":    total_patients,
        "total_scans":       total_scans,
        "positive_cases":    positive,
        "positive_percent":  round(positive / total_scans * 100, 1) if total_scans else 0,
        "total_doctors":     total_doctors,
        "total_predictions": total_scans,
        "today_predictions": today_count,
        "role":              role,
    }


async def _safe_count(db, coll, query=None):
    try:
        return await asyncio.wait_for(
            db[coll].count_documents(query or {}), timeout=QUERY_TIMEOUT
        )
    except Exception as e:
        print(f"⚠️ Count failed on {coll}: {type(e).__name__}")
        return 0


async def _name_maps(db):
    """patient id/code -> name, doctor id -> name (small fields only)."""
    pmap, dmap = {}, {}
    try:
        patients = await asyncio.wait_for(
            db["patients"].find({}, {"full_name": 1, "name": 1, "patient_id": 1}).to_list(length=2000),
            timeout=QUERY_TIMEOUT
        )
        for p in patients:
            name = p.get("full_name") or p.get("name") or "—"
            pmap[str(p["_id"])] = name                  # results that store the MongoDB _id
            if p.get("patient_id"):
                pmap[str(p["patient_id"])] = name       # results that store the code (P001)
    except Exception as e:
        print(f"⚠️ Patient lookup failed: {type(e).__name__}")

    try:
        doctors = await asyncio.wait_for(
            db["users"].find({"role": {"$in": ["doctor", "admin"]}}, {"full_name": 1}).to_list(length=200),
            timeout=QUERY_TIMEOUT
        )
        dmap = {str(d["_id"]): d.get("full_name", "—") for d in doctors}
    except Exception as e:
        print(f"⚠️ Doctor lookup failed: {type(e).__name__}")
    return pmap, dmap


# ── COMBINED ENDPOINT (use this from the frontend) ─────────────
@router.get("/overview")
async def dashboard_overview(user=Depends(get_current_user)):
    """
    Everything the dashboard needs in ONE request:
    stats, weekly volume, disease distribution, and recent activity.
    """
    db = get_database()

    # all independent → run at the same time
    total_patients, total_doctors, results, (pmap, dmap) = await asyncio.gather(
        _safe_count(db, "patients"),
        _safe_count(db, "users", {"role": "doctor"}),
        _gather_all_results(db),
        _name_maps(db),
    )

    recent = sorted(results, key=lambda x: x.get("created_at") or "", reverse=True)[:6]
    recent = [
        {
            **r,
            "patient_name": pmap.get(str(r.get("patient_id")), "—"),
            "doctor_name":  dmap.get(str(r.get("doctor_id")), "—"),
            "date":         r.get("created_at"),
        }
        for r in recent
    ]

    return {
        "stats":        _build_stats(results, total_patients, total_doctors, user["role"]),
        "weekly":       _build_weekly(results),
        "distribution": _build_distribution(results),
        "recent":       recent,
    }


# ── Individual endpoints (kept for compatibility / testing) ────
@router.get("/stats")
async def dashboard_stats(user=Depends(get_current_user)):
    db = get_database()
    total_patients = await _safe_count(db, "patients")
    total_doctors  = await _safe_count(db, "users", {"role": "doctor"})
    results        = await _gather_all_results(db)
    return _build_stats(results, total_patients, total_doctors, user["role"])


@router.get("/weekly-volume")
async def weekly_volume(user=Depends(get_current_user)):
    db = get_database()
    results = await _gather_all_results(db)
    return _build_weekly(results)


@router.get("/disease-distribution")
async def disease_distribution(user=Depends(get_current_user)):
    db = get_database()
    results = await _gather_all_results(db)
    return _build_distribution(results)