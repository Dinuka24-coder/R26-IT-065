from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
from collections import defaultdict
from app.core.dependencies import get_current_user
from app.database import get_database

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

COMPONENT_COLLECTIONS = {
    "pneumothorax_results": ("Pneumothorax", "X-ray"),
    "pneumonia_results":    ("Pneumonia",    "X-ray"),
    "tuberculosis_results": ("Tuberculosis", "X-ray"),
    "lungcancer_results":   ("Lung Cancer",  "CT Scan"),
}


async def _gather_all_results(db):
    records = []

    # Only query collections that actually exist
    existing = await db.list_collection_names()

    for coll, (disease, scan) in COMPONENT_COLLECTIONS.items():
        if coll not in existing:
            continue                     # ← skip missing collections

        try:
            cursor = db[coll].find()
            async for r in cursor:
                positive = "Detected" in str(r.get("prediction", ""))
                records.append({
                    "patient_id": r.get("patient_id"),
                    "doctor_id":  r.get("doctor_id"),
                    "disease":    disease if positive else "Normal",
                    "scan_type":  scan,
                    "confidence": r.get("confidence", 0),
                    "status":     "Positive" if positive else "Normal",
                    "created_at": r.get("created_at"),
                })
        except Exception as e:
            print(f"⚠️ Skipping {coll}: {e}")
            continue                     # ← never crash the whole endpoint

    return records


@router.get("/stats")
async def dashboard_stats(user=Depends(get_current_user)):
    db = get_database()

    total_patients = await db["patients"].count_documents({})
    total_doctors  = await db["users"].count_documents({"role": "doctor"})
    results        = await _gather_all_results(db)

    total_scans = len(results)
    positive    = sum(1 for r in results if r["status"] == "Positive")

    # today's predictions
    today = datetime.utcnow().date().isoformat()
    today_count = sum(1 for r in results if str(r.get("created_at", "")).startswith(today))

    return {
        "total_patients":    total_patients,
        "total_scans":       total_scans,
        "positive_cases":    positive,
        "positive_percent":  round(positive / total_scans * 100, 1) if total_scans else 0,
        "total_doctors":     total_doctors,
        "total_predictions": total_scans,
        "today_predictions": today_count,
        "role":              user["role"],
    }


@router.get("/weekly-volume")
async def weekly_volume(user=Depends(get_current_user)):
    """X-ray vs CT scan counts for the last 7 days."""
    db = get_database()
    results = await _gather_all_results(db)

    days = [(datetime.utcnow().date() - timedelta(days=i)) for i in range(6, -1, -1)]
    buckets = {d.isoformat(): {"xray": 0, "ct": 0} for d in days}

    for r in results:
        day = str(r.get("created_at", ""))[:10]
        if day in buckets:
            if r["scan_type"] == "CT Scan":
                buckets[day]["ct"] += 1
            else:
                buckets[day]["xray"] += 1

    return [
        {
            "day":  datetime.fromisoformat(d).strftime("%a"),
            "xray": buckets[d]["xray"],
            "ct":   buckets[d]["ct"],
        }
        for d in buckets
    ]


@router.get("/disease-distribution")
async def disease_distribution(user=Depends(get_current_user)):
    db = get_database()
    results = await _gather_all_results(db)

    counts = defaultdict(int)
    for r in results:
        counts[r["disease"]] += 1

    total = sum(counts.values()) or 1
    colors = {
        "Pneumonia": "#3b82f6", "Pneumothorax": "#ef4444",
        "Tuberculosis": "#f59e0b", "Lung Cancer": "#8b5cf6", "Normal": "#22c55e",
    }

    return [
        {"name": k, "value": v, "percent": round(v / total * 100), "color": colors.get(k, "#94a3b8")}
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ]