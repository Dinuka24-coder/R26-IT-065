import asyncio
from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
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

QUERY_TIMEOUT = 10.0
MAX_DOCS      = 500

POSITIVE_MARKERS = ("detected", "positive", "malignant", "abnormal")
NEGATIVE_MARKERS = ("normal", "no ", "negative", "clear")

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
        return r["created_at"]
    ts = r.get("timestamp")
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts) if ts else None

# ── Shared helpers ─────────────────────────────────────────────
async def _gather_all_results(db):
    """Collect prediction results from all component collections."""
    records = []

    for coll, (disease, scan) in COMPONENT_COLLECTIONS.items():
        if not collection_exists(coll):        # instant — no DB round trip
            continue

        try:
            docs = await asyncio.wait_for(
                db[coll].find().to_list(length=MAX_DOCS),
                timeout=QUERY_TIMEOUT
            )
        except Exception as e:
            print(f"⚠️ Skipping {coll}: {type(e).__name__}")
            continue

        for r in docs:
            positive = _is_positive(r)
            records.append({
                "patient_id": r.get("patient_id"),
                "doctor_id":  r.get("doctor_id"),
                "disease":    disease if positive else "Normal",
                "scan_type":  scan,
                "confidence": r.get("confidence", 0),
                "status":     "Positive" if positive else "Normal",
                "created_at": _get_created_at(r),
            })

    return records


def _build_weekly(results):
    days = [(datetime.utcnow().date() - timedelta(days=i)) for i in range(6, -1, -1)]
    buckets = {d.isoformat(): {"xray": 0, "ct": 0} for d in days}

    for r in results:
        day = str(r.get("created_at", ""))[:10]
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
    today       = datetime.utcnow().date().isoformat()
    today_count = sum(1 for r in results if str(r.get("created_at", "")).startswith(today))

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


# ── COMBINED ENDPOINT (use this from the frontend) ─────────────
@router.get("/overview")
async def dashboard_overview(user=Depends(get_current_user)):
    """
    Everything the dashboard needs in ONE request:
    stats, weekly volume, disease distribution, and recent activity.
    """
    db = get_database()

    total_patients = await _safe_count(db, "patients")
    total_doctors  = await _safe_count(db, "users", {"role": "doctor"})
    results        = await _gather_all_results(db)

    # Name lookups for recent activity
    pmap, dmap = {}, {}
    try:
        patients = await asyncio.wait_for(
            db["patients"].find().to_list(length=MAX_DOCS), timeout=QUERY_TIMEOUT
        )
        pmap = {str(p["_id"]): p.get("full_name", "—") for p in patients}
    except Exception as e:
        print(f"⚠️ Patient lookup failed: {type(e).__name__}")

    try:
        doctors = await asyncio.wait_for(
            db["users"].find({"role": {"$in": ["doctor", "admin"]}}).to_list(length=200),
            timeout=QUERY_TIMEOUT
        )
        dmap = {str(d["_id"]): d.get("full_name", "—") for d in doctors}
    except Exception as e:
        print(f"⚠️ Doctor lookup failed: {type(e).__name__}")

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