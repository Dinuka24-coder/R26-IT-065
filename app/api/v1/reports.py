from fastapi import APIRouter, Depends
from bson import ObjectId
from app.core.dependencies import require_doctor_or_admin
from app.database import get_database

router = APIRouter(prefix="/reports", tags=["Reports"])

COMPONENT_COLLECTIONS = {
    "pneumothorax_results": ("Pneumothorax", "X-ray"),
    "pneumonia_results":    ("Pneumonia",    "X-ray"),
    "tuberculosis_results": ("Tuberculosis", "X-ray"),
    "lungcancer_results":   ("Lung Cancer",  "CT Scan"),
}


@router.get("/history")
async def history_report(patient_id: str = None, doctor_id: str = None,
                         user=Depends(require_doctor_or_admin)):
    db = get_database()
    existing = await db.list_collection_names()

    # Build name lookups
    patient_map = {}
    async for p in db["patients"].find():
        patient_map[str(p["_id"])] = p.get("full_name", "Unknown")

    doctor_map = {}
    async for d in db["users"].find({"role": {"$in": ["doctor", "admin"]}}):
        doctor_map[str(d["_id"])] = d.get("full_name", "Unknown")

    records = []
    for coll, (disease, scan) in COMPONENT_COLLECTIONS.items():
        if coll not in existing:
            continue

        query = {}
        if patient_id: query["patient_id"] = patient_id
        if doctor_id:  query["doctor_id"]  = doctor_id

        try:
            cursor = db[coll].find(query)
            async for r in cursor:
                positive = "Detected" in str(r.get("prediction", ""))
                records.append({
                    "patient_id":   r.get("patient_id"),
                    "patient_name": patient_map.get(str(r.get("patient_id")), "—"),
                    "doctor_id":    r.get("doctor_id"),
                    "doctor_name":  doctor_map.get(str(r.get("doctor_id")), "—"),
                    "date":         r.get("created_at"),
                    "disease":      disease if positive else "Normal",
                    "scan_type":    scan,
                    "confidence":   r.get("confidence"),
                    "status":       "Positive" if positive else "Normal",
                })
        except Exception as e:
            print(f"⚠️ Skipping {coll}: {e}")
            continue

    records.sort(key=lambda x: x["date"] or "", reverse=True)
    return {"total": len(records), "records": records}