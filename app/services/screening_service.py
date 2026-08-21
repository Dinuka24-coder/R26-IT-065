import asyncio
import time
import cv2
import numpy as np
from datetime import datetime

from app.ml_models.component1.xray_validator import is_xray
from app.ml_models.component2.inference import run_pneumonia_inference, InvalidXRayError
from app.services.comp1_service import run_prediction as run_pneumothorax
from app.services.comp2_service import save_pneumonia_prediction
from app.database import get_database


# ══════════════════════════════════════════════════════════════
#  PNEUMONIA ADAPTER
#  Component 2's engine is synchronous, takes a decoded cv2 image,
#  returns a tuple, and uses different field names. Normalized here.
#  Its own OOD and TB shields stay ACTIVE — they catch cases the
#  central validator does not, and a rejection is surfaced as a
#  result rather than a pipeline failure.
# ══════════════════════════════════════════════════════════════

def _map_severity(severity) -> str:
    """Map the pneumonia severity label onto the shared urgency scale."""
    if not severity:
        return "Low"
    s = str(severity).lower()
    if any(w in s for w in ("severe", "critical", "high")):
        return "High"
    if any(w in s for w in ("moderate", "medium")):
        return "Moderate"
    return "Low"


async def run_pneumonia_adapted(patient_id: str, image_bytes: bytes, user: dict = None):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image for pneumonia analysis")

    # Component 2's own OOD + TB shields run here.
    # May raise InvalidXRayError — handled in _run_engine().
    diagnosis, confidence, severity, heatmap_b64, heatmap_sev = await asyncio.to_thread(
        run_pneumonia_inference, img
    )

    prediction = str(diagnosis).title()
    if prediction.upper() == "PNEUMONIA":
        prediction = "Pneumonia Detected"

    try:
        await save_pneumonia_prediction(
            patient_id=patient_id,
            filename="full_screening",
            diagnosis=diagnosis,
            confidence=confidence,
            severity=severity,
            heatmap_base64=heatmap_b64,
            affected_area_percent=(heatmap_sev or {}).get("affected_area_percent"),
            mean_intensity=(heatmap_sev or {}).get("mean_intensity"),
            doctor_id=str(user["_id"]) if user else None,
        )
    except Exception as e:
        print(f"      ⚠️  pneumonia record not saved: {type(e).__name__}")

    return {
        "prediction":            prediction,
        "confidence":            round(float(confidence), 2),
        "urgency":               _map_severity(severity),
        "heatmap_base64":        heatmap_b64,
        "severity":              severity,
        "affected_area_percent": (heatmap_sev or {}).get("affected_area_percent"),
    }


# ══════════════════════════════════════════════════════════════
#  ENGINE REGISTRY
# ══════════════════════════════════════════════════════════════

XRAY_ENGINES = {
    "pneumothorax": {
        "label":  "Pneumothorax",
        "runner": run_pneumothorax,
        "ready":  True,
    },
    "pneumonia": {
        "label":  "Pneumonia",
        "runner": run_pneumonia_adapted,
        "ready":  True,
    },
    "tuberculosis": {
        "label":  "Tuberculosis",
        "runner": None,          # TODO: connect when component 3 is ready
        "ready":  False,
    },
}

DETAIL_KEYS = (
    "affected_lung_pct", "pleural_separation", "segmented_area_pct",   # component 1
    "severity", "affected_area_percent",                               # component 2
)

ENGINE_TIMEOUT = 60.0


# ══════════════════════════════════════════════════════════════
#  ENGINE RUNNER
# ══════════════════════════════════════════════════════════════

async def _run_engine(key: str, cfg: dict, patient_id: str, image_bytes: bytes, user: dict):
    """Run one engine. Never raises — always returns a result dict."""
    label = cfg["label"]

    if not cfg["ready"] or cfg["runner"] is None:
        print(f"   ⏭️  {label:<14} SKIPPED — engine not connected")
        return {
            "component": key, "label": label, "available": False,
            "message": f"{label} engine is not connected yet.",
        }

    t = time.time()
    print(f"   ▶️  {label:<14} running…")

    try:
        result = await asyncio.wait_for(
            cfg["runner"](patient_id, image_bytes, user),
            timeout=ENGINE_TIMEOUT
        )

        detected = "Detected" in str(result.get("prediction", ""))
        conf     = result.get("confidence", 0)
        urgency  = result.get("urgency", "Low")
        elapsed  = time.time() - t

        icon = "🔴" if detected else "🟢"
        print(f"   {icon} {label:<14} {result.get('prediction')}")
        print(f"      confidence {conf}%  ·  urgency {urgency}  ·  {elapsed:.2f}s")

        details = {k: v for k, v in result.items() if k in DETAIL_KEYS and v is not None}
        if details:
            print(f"      details: {details}")
        if detected and not result.get("heatmap_base64"):
            print(f"      (no heatmap returned)")

        return {
            "component":      key,
            "label":          label,
            "available":      True,
            "detected":       detected,
            "prediction":     result.get("prediction"),
            "confidence":     conf,
            "urgency":        urgency,
            "heatmap_base64": result.get("heatmap_base64"),
            "details":        details,
            "duration_sec":   round(elapsed, 2),
        }

    except InvalidXRayError as e:
        # The engine's own validator rejected the image.
        # Reported as a result, not a failure — the other engines still run.
        elapsed = time.time() - t
        print(f"   ⚠️  {label:<14} REJECTED by its own validator  [{elapsed:.2f}s]")
        print(f"      reason: {e}")
        return {
            "component":    key,
            "label":        label,
            "available":    True,
            "detected":     False,
            "prediction":   "Not suitable for this engine",
            "confidence":   0,
            "urgency":      "Low",
            "rejected":     True,
            "message":      str(e),
            "duration_sec": round(elapsed, 2),
        }

    except asyncio.TimeoutError:
        print(f"   ⏰ {label:<14} TIMED OUT after {ENGINE_TIMEOUT}s")
        return {"component": key, "label": label, "available": False,
                "message": f"{label} analysis timed out."}

    except Exception as e:
        elapsed = time.time() - t
        print(f"   ❌ {label:<14} FAILED — {type(e).__name__}: {e}  [{elapsed:.2f}s]")
        import traceback
        traceback.print_exc()
        return {"component": key, "label": label, "available": False,
                "message": f"{label} analysis failed: {type(e).__name__}"}


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

async def run_full_screening(patient_id: str, image_bytes: bytes, user: dict = None):
    """
    Run one chest X-ray through all available X-ray engines in parallel.

    Two layers of validation:
      1. Central chest X-ray validator — gates the whole pipeline
      2. Each engine's own shield — may reject individually; reported per engine
    """
    t_start = time.time()

    print("\n" + "═" * 64)
    print(f"🔬 FULL SCREENING  |  patient {patient_id}")
    print(f"   image {len(image_bytes)/1024:.1f} KB  ·  "
          f"doctor {str(user['_id']) if user else 'n/a'}")
    print("═" * 64)

    # ── GATE 1: CENTRAL VALIDATOR ──────────────────────────────
    t0 = time.time()
    validation = is_xray(image_bytes)

    print(f"\n📋 GATE 1 — Central chest X-ray validator  ({time.time()-t0:.2f}s)")
    print(f"   aspect ratio : {validation.get('aspect_ratio', 'n/a')}")
    print(f"   distance     : {validation.get('distance', 'n/a')}  "
          f"(threshold {validation.get('threshold', 'n/a')})")

    if validation.get("features"):
        print("   features:")
        for k, v in validation["features"].items():
            print(f"      {k:<22} {v}")

    if validation.get("deviations"):
        worst = max(validation["deviations"], key=validation["deviations"].get)
        print(f"   largest deviation: {worst} (Δ {validation['deviations'][worst]})")

    accepted = validation.get("is_xray")
    print(f"   → {'✅ ACCEPTED' if accepted else '❌ REJECTED'}")

    if not accepted:
        reason = validation.get("reason") or validation.get("error") \
                 or "Image does not match a chest X-ray profile."
        print(f"   reason: {reason}")
        print("═" * 64 + "\n")
        return {"status": "rejected", "error": reason, "validation": validation}

    # ── GATE 2: PER-ENGINE SHIELDS + INFERENCE ─────────────────
    ready = [c["label"] for c in XRAY_ENGINES.values() if c["ready"]]
    print(f"\n⚙️  GATE 2 — Engines  ({len(ready)} of {len(XRAY_ENGINES)} connected: {', '.join(ready)})")
    print(f"   each engine applies its own validation before inference")

    t1 = time.time()
    tasks = [_run_engine(k, cfg, patient_id, image_bytes, user)
             for k, cfg in XRAY_ENGINES.items()]
    results = await asyncio.gather(*tasks)
    engine_time = time.time() - t1

    # ── SUMMARY ────────────────────────────────────────────────
    completed = [r for r in results if r.get("available")]
    analysed  = [r for r in completed if not r.get("rejected")]
    rejected  = [r for r in completed if r.get("rejected")]
    findings  = [r for r in analysed if r.get("detected")]
    findings.sort(key=lambda r: r.get("confidence", 0), reverse=True)

    urgency_rank = {"High": 3, "Moderate": 2, "Low": 1}
    overall_urgency = max(
        (f.get("urgency", "Low") for f in findings),
        key=lambda u: urgency_rank.get(u, 0),
        default="Low",
    )

    if not analysed:
        verdict = "No engine could analyse this scan"
    elif findings:
        verdict = ", ".join(f["label"] for f in findings)
    else:
        verdict = "No abnormality detected"

    screening = {
        "status":           "success",
        "patient_id":       patient_id,
        "doctor_id":        str(user["_id"]) if user else None,
        "scan_type":        "X-ray",
        "screening_type":   "full",
        "engines_run":      len(analysed),
        "engines_rejected": len(rejected),
        "engines_total":    len(XRAY_ENGINES),
        "findings_count":   len(findings),
        "verdict":          verdict,
        "overall_urgency":  overall_urgency,
        "primary_finding":  findings[0]["label"] if findings else None,
        "results":          results,
        "created_at":       datetime.utcnow().isoformat(),
    }

    print(f"\n📊 SUMMARY")
    print(f"   analysed    : {len(analysed)}/{len(XRAY_ENGINES)}  ({engine_time:.2f}s parallel)")
    if rejected:
        print(f"   rejected    : {len(rejected)}  ({', '.join(r['label'] for r in rejected)})")
    print(f"   findings    : {len(findings)}")
    print(f"   verdict     : {verdict}")
    print(f"   urgency     : {overall_urgency}")
    if findings:
        for f in findings:
            print(f"      • {f['label']}  {f['confidence']}%  ({f['urgency']})")
    print(f"   total time  : {time.time()-t_start:.2f}s")

    # ── SAVE SUMMARY RECORD (heatmaps stripped — too large) ────
    try:
        db = get_database()
        record = {k: v for k, v in screening.items() if k != "results"}
        record["summary"] = [
            {k: v for k, v in r.items() if k != "heatmap_base64"} for r in results
        ]
        await db["screening_results"].insert_one(record)
        print(f"   saved to screening_results ✅")
    except Exception as e:
        print(f"   ⚠️  screening record not saved: {type(e).__name__}")

    print("═" * 64 + "\n")
    return screening