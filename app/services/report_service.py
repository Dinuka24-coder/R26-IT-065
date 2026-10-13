"""
PDF report generation for PulmoAI (ReportLab).

Reports:
  1. Summary report  – filtered results from all four components
                       (becomes a Doctor / Disease / Patient report when that filter is set)
  2. Result report   – one prediction for one patient
  3. Full screening  – one X-ray, several disease results

Data access (MongoDB) and PDF drawing are kept in separate functions,
so the PDF part can be tested without a database.
"""

from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from xml.sax.saxutils import escape
from typing import Optional

from bson import ObjectId
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# collection -> (component key, disease name, scan type)
COMPONENTS = {
    "pneumothorax_results": ("pneumothorax", "Pneumothorax", "X-ray"),
    "pneumonia_results":    ("pneumonia",    "Pneumonia",    "X-ray"),
    "tuberculosis_results": ("tuberculosis", "Tuberculosis", "X-ray"),
    "lung_cancer_results":  ("lungcancer",   "Lung Cancer",  "CT Scan"),
}
COMPONENT_TO_COLLECTION = {v[0]: k for k, v in COMPONENTS.items()}

MAX_ROWS = 1000

# Fields read for the result list / summary report (keep this small = fast)
LIST_FIELDS = {"patient_id": 1, "doctor_id": 1, "prediction": 1, "confidence": 1,
               "urgency": 1, "severity": 1, "created_at": 1,
               # older / standalone TB records use these names instead
               "diagnosis": 1, "confidence_score": 1, "timestamp": 1,
               # used to tell Full Screening results apart from single analyses
               "filename": 1, "source": 1}

# Times in reports are shown in Sri Lanka time (UTC+05:30, no daylight saving)
LOCAL_TZ = timezone(timedelta(hours=5, minutes=30))

# Fields hidden from the Details table when they are 0 / empty
HIDE_WHEN_EMPTY = {"segmented_area_pct", "affected_lung_pct", "boundary_length_pct"}

# Stored fields that are not useful in the PDF (internal or duplicates of fields already shown)
HIDE_IN_PDF = {"explanation_image", "filename", "input_type", "diagnosis", "confidence_score",
               "timestamp", "bounding_box", "raw_score"}
SKIP_FIELDS = {"_id", "heatmap_base64", "heatmap", "image_base64", "original_base64", "standard_heatmap_base64","heatmap_url", "heatmap_only_url", "densenet_gradcam_url", "densenet_heatmap_only_url",
               "custom_gradcam_url", "custom_heatmap_only_url", "combined_gradcam_url"}
FIELD_LABELS = {
    "affected_lung_pct": "Attention Coverage (%)",
    "boundary_length_pct": "Boundary Extent (%)",
    "pleural_separation": "Pleural Separation",
    "raw_score": "Raw model score",
    "affected_area_percent": "Affected Area (%)",
    "mean_intensity": "Mean Attention Intensity",
    "clinical_note": "Clinical Note",
    "urgency": "Urgency",
    "severity": "Severity",
    "class_probabilities": "Class probabilities",
    "input_check": "Input check",
    "comparison_metrics": "Grad-CAM branch agreement",
}

# colours
INK = colors.HexColor("#15242F")
MUTED = colors.HexColor("#566874")
LINE = colors.HexColor("#D2DCE1")
HEAD_BG = colors.HexColor("#0F1E29")
ROW_ALT = colors.HexColor("#F3F7F9")
POS = colors.HexColor("#B42318")
NEG = colors.HexColor("#15803D")

DISCLAIMER = (
    "AI results must be reviewed and confirmed by a qualified clinician "
    "and do not replace clinical diagnosis."
)


# ── small helpers ────────────────────────────────────────────────
def to_datetime(value) -> Optional[datetime]:
    """created_at may be a datetime or an ISO string (with or without timezone)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def fmt_dt(value) -> str:
    dt = to_datetime(value)
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p") if dt else "-"


def now_local() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p")


def _patient_query(pid: str) -> dict:
    """Results may store the patient's code (e.g. P011) or the patient's MongoDB _id."""
    ors = [{"patient_id": pid}]
    if ObjectId.is_valid(pid):
        ors.append({"_id": ObjectId(pid)})
    return {"$or": ors}

NEGATIVE_WORDS = ("normal", "no ", "not ", "negative", "clear", "other abnormality", "healthy", "non-tb")
POSITIVE_WORDS = ("detected", "positive", "malignant", "carcinoma", "nodule", "tuberculosis")


def is_positive(prediction) -> bool:
    """Works for every component's wording, e.g. 'Pneumonia Detected', 'No Pneumothorax',
    'Other Abnormality' (TB: not TB), 'adenocarcinoma' / 'normal' (lung cancer classes)."""
    p = str(prediction or "").lower().strip()
    if not p or any(w in p for w in NEGATIVE_WORDS):
        return False
    return any(w in p for w in POSITIVE_WORDS)


def pretty_prediction(prediction) -> str:
    """'large.cell.carcinoma' -> 'Large Cell Carcinoma'; other text unchanged."""
    p = str(prediction or "-").strip()
    if p.islower() or "." in p or "_" in p:
        return p.replace(".", " ").replace("_", " ").title()
    return p


def fmt_conf(value) -> str:
    """99.97 -> '99.97%', 80.3 -> '80.3%' (never rounds 99.97 up to 100)."""
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.2f}".rstrip("0").rstrip(".") + "%"


# TB model labels -> display text (same as screening_service.TB_DISPLAY)
TB_DISPLAY = {"Healthy": "Normal", "Non-TB": "Other Abnormality", "Tuberculosis": "Tuberculosis Detected"}


def normalize_result(r: dict) -> dict:
    """Fill the shared field names when a record only has the old TB names
    (diagnosis / confidence_score / timestamp). Returns the same dict."""
    if not r.get("prediction") and r.get("diagnosis"):
        r["prediction"] = TB_DISPLAY.get(r["diagnosis"], r["diagnosis"])
    if r.get("confidence") is None and r.get("confidence_score") is not None:
        r["confidence"] = r["confidence_score"]
    if not r.get("created_at") and r.get("timestamp"):
        r["created_at"] = r["timestamp"]
    return r


def user_id(user: dict) -> str:
    return str(user.get("_id") or user.get("id") or user.get("user_id") or "")


def fmt_value(value) -> str:
    if isinstance(value, bool):
        return "Present" if value else "Absent"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt_value(v) for v in value[:8])
    if isinstance(value, dict):
        return ", ".join(f"{escape(str(k))}: {fmt_value(v)}" for k, v in list(value.items())[:8])
    return escape(str(value))

def _fmt_class_probabilities(d: dict) -> str:
    """{'adenocarcinoma': 0.61, ...} -> 'Adenocarcinoma: 61%, ...' (highest first)."""
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{str(k).replace('.', ' ').title()}: {float(v) * 100:.0f}%" for k, v in items)


def _fmt_input_check(d: dict) -> str:
    """Suitability check stored by Component 4."""
    ok = "Passed" if d["within_distribution"] else "Failed"
    extra = ""
    if d.get("distance") is not None and d.get("threshold") is not None:
        extra = f" (distance {float(d['distance']):.2f}, threshold {float(d['threshold']):.2f})"
    return ok + extra


def _fmt_comparison_metrics(d: dict) -> str:
    """Plain-language version of the DenseNet121 / Custom CNN agreement values."""
    text = (f"Overlap between branches (IoU): {d['iou_at_0.5']:.2f}, "
            f"DenseNet121 highlighted area: {d['dense_area_above_0.5'] * 100:.0f}%, "
            f"Custom CNN highlighted area: {d['custom_area_above_0.5'] * 100:.0f}%, "
            f"Centre distance: {d['center_of_mass_distance_px']:.1f} px")
    return (text + "<br/><font size='7' color='#566874'>Technical Grad-CAM agreement values. "
            "They do not indicate clinical accuracy or prove tumour localization.</font>")


# value formatters for dict fields stored by Component 4 (fall back to fmt_value if the shape differs)
SPECIAL_FORMATTERS = {
    "class_probabilities": _fmt_class_probabilities,
    "input_check": _fmt_input_check,
    "comparison_metrics": _fmt_comparison_metrics,
}


def fmt_field(key: str, value) -> str:
    fmt = SPECIAL_FORMATTERS.get(key)
    if fmt and isinstance(value, dict):
        try:
            return fmt(value)
        except (KeyError, TypeError, ValueError):
            pass
    return fmt_value(value)


# ── data access (MongoDB / Motor) ────────────────────────────────
async def gather_results(
    db,
    user: dict,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    disease: Optional[str] = None,
    status: Optional[str] = None,
    patient_id: Optional[str] = None,
    doctor_id: Optional[str] = None,
    source: Optional[str] = None,
) -> list:
    """Collect results from all component collections, apply filters, newest first.
    source: 'all' | 'single' (single analysis pages) | 'full_screening'"""
    existing = set(await db.list_collection_names())
    screening_of = await screening_map(db) if "screening_results" in existing else {}

    query = {}
    if patient_id:
        # results may store the patient code (P001) or the patient's MongoDB _id
        ids = {str(patient_id)}
        p = await db["patients"].find_one(_patient_query(str(patient_id)))
        if p:
            ids |= {str(p.get("_id")), str(p.get("patient_id") or "")}
        query["patient_id"] = {"$in": [i for i in ids if i and i != "None"]}

    def doctor_filter(uid: str):
        # doctor_id may be saved as a string or as an ObjectId
        return {"$in": [uid, ObjectId(uid)]} if ObjectId.is_valid(uid) else uid

    if user.get("role") != "admin":
        query["doctor_id"] = doctor_filter(user_id(user))   # doctors only see their own results
    elif doctor_id:
        query["doctor_id"] = doctor_filter(doctor_id)

    start = datetime.combine(start_date, time.min, tzinfo=LOCAL_TZ) if start_date else None
    end = datetime.combine(end_date, time.max, tzinfo=LOCAL_TZ) if end_date else None

    rows = []
    for coll, (component, disease_name, scan) in COMPONENTS.items():
        if coll not in existing:
            continue
        if disease and disease.lower() not in ("all", disease_name.lower(), component):
            continue
        # Only fetch the small fields the list needs. Heatmaps / images are big and
        # made the query time out on MongoDB Atlas.
        async for r in db[coll].find(query, LIST_FIELDS).sort("_id", -1).limit(MAX_ROWS):
            normalize_result(r)
            created = to_datetime(r.get("created_at"))
            if start and (created is None or created < start):
                continue
            if end and (created is None or created > end):
                continue
            rid = str(r.get("_id"))
            screening_id = screening_of.get(rid) or r.get("screening_id")
            from_screening = bool(screening_id) or r.get("source") == "full_screening" \
                or r.get("filename") == "full_screening"
            if source == "single" and from_screening:
                continue
            if source == "full_screening" and not from_screening:
                continue
            positive = is_positive(r.get("prediction"))
            if status and status.lower() != "all":
                if (status.lower() == "positive") != positive:
                    continue
            rows.append({
                "result_id": str(r.get("_id")),
                "component": component,
                "disease": disease_name,
                "scan_type": scan,
                "patient_id": str(r.get("patient_id", "-")),
                "doctor_id": str(r.get("doctor_id", "")),
                "prediction": pretty_prediction(r.get("prediction")),
                "confidence": r.get("confidence"),
                "urgency": r.get("urgency") or r.get("severity"),
                "positive": positive,
                "created_at": created,
                "source": "full_screening" if from_screening else "single",
                "screening_id": str(screening_id) if screening_id else None,
            })

    rows.sort(key=lambda x: x["created_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return rows[:MAX_ROWS]


async def screening_map(db) -> dict:
    """result_id -> screening_id, read from the Full Screening records."""
    out = {}
    cursor = db["screening_results"].find({}, {"summary.result_id": 1}).sort("_id", -1).limit(MAX_ROWS)
    async for s in cursor:
        for e in s.get("summary") or []:
            if e.get("result_id"):
                out[str(e["result_id"])] = str(s["_id"])
    return out


async def attach_names(db, rows: list) -> list:
    """Adds patient_code, patient_name and doctor_name to every row (in place)."""
    patient_ids = {r["patient_id"] for r in rows if r.get("patient_id")}
    info = {}
    if patient_ids:
        oids = [ObjectId(x) for x in patient_ids if ObjectId.is_valid(x)]
        query = {"$or": [{"patient_id": {"$in": list(patient_ids)}}, {"_id": {"$in": oids}}]}
        async for p in db["patients"].find(query):
            entry = {"code": p.get("patient_id") or str(p.get("_id")),
                     "name": p.get("full_name") or p.get("name") or ""}
            for key in (p.get("patient_id"), str(p.get("_id"))):
                if key:
                    info[key] = entry

    doctors = {}
    oids = [ObjectId(d) for d in {r["doctor_id"] for r in rows if r.get("doctor_id")} if ObjectId.is_valid(d)]
    if oids:
        async for u in db["users"].find({"_id": {"$in": oids}}):
            doctors[str(u["_id"])] = u.get("full_name") or u.get("email") or ""

    for r in rows:
        p = info.get(r["patient_id"], {})
        r["patient_code"] = p.get("code") or r["patient_id"]
        r["patient_name"] = p.get("name", "")
        r["doctor_name"] = doctors.get(r["doctor_id"], "")
    return rows


def search_rows(rows: list, text: Optional[str]) -> list:
    """Keep rows where the text appears in patient, doctor, disease or result."""
    q = (text or "").strip().lower()
    if not q:
        return rows
    fields = ("patient_code", "patient_name", "doctor_name", "disease", "prediction")
    return [r for r in rows if any(q in str(r.get(f, "")).lower() for f in fields)]


async def get_result(db, component: str, result_id: str) -> Optional[dict]:
    coll = COMPONENT_TO_COLLECTION.get(component)
    if not coll or not ObjectId.is_valid(result_id):
        return None
    r = await db[coll].find_one({"_id": ObjectId(result_id)})
    return normalize_result(r) if r else None


async def get_patient(db, patient_id: str) -> Optional[dict]:
    if not patient_id:
        return None
    return await db["patients"].find_one(_patient_query(str(patient_id)))


async def get_user_name(db, uid: str) -> str:
    if not uid or not ObjectId.is_valid(uid):
        return ""
    u = await db["users"].find_one({"_id": ObjectId(uid)})
    return (u or {}).get("full_name") or (u or {}).get("email") or ""


# ── PDF building blocks ──────────────────────────────────────────
def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17,
                                textColor=INK, alignment=0, spaceAfter=2, leading=21),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9, textColor=MUTED, leading=12),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                             textColor=INK, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, textColor=INK, leading=13),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.5, textColor=INK, leading=10.5),
        "cellr": ParagraphStyle("cr", parent=ss["Normal"], fontSize=8.5, textColor=INK, leading=10.5,
                                alignment=TA_RIGHT),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.8, textColor=MUTED, leading=10),
        "big": ParagraphStyle("big", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=15, leading=19),
    }


def _header_footer(title: str):
    def draw(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(HEAD_BG)
        canvas.rect(0, h - 14 * mm, w, 14 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10.5)
        canvas.drawString(15 * mm, h - 9 * mm, "PulmoAI")
        canvas.setFont("Helvetica", 8.5)
        canvas.drawString(33 * mm, h - 9 * mm, "Multi-Modal Pulmonary Disease Detection System")
        canvas.drawRightString(w - 15 * mm, h - 9 * mm, title)
        canvas.setStrokeColor(LINE)
        canvas.line(15 * mm, 13 * mm, w - 15 * mm, 13 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(15 * mm, 8.5 * mm, "AI decision support only - confirm with clinical assessment.")
        canvas.drawRightString(w - 15 * mm, 8.5 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, title=title, author="PulmoAI",
                             leftMargin=15 * mm, rightMargin=15 * mm,
                             topMargin=22 * mm, bottomMargin=18 * mm)


def _kv_table(pairs, st, col1=45 * mm, col2=None):
    data = [[Paragraph(f"<font color='#566874'>{k}</font>", st["cell"]), Paragraph(str(v), st["cell"])]
            for k, v in pairs]
    t = Table(data, colWidths=[col1, col2 or (180 * mm - col1)])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _date_range(start, end) -> str:
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"From {start}"
    if end:
        return f"Up to {end}"
    return "All dates"


# ── 1. summary report ────────────────────────────────────────────
def build_summary_pdf(rows: list, filters: dict, generated_by: str) -> bytes:
    """rows must already have names (see attach_names)."""
    title = filters.get("title") or "Diagnostic Summary Report"
    st = _styles()
    buf = BytesIO()
    doc = _doc(buf, f"PulmoAI {title}")
    story = []

    story.append(Paragraph(title, st["title"]))
    story.append(Paragraph(f"Generated {now_local()} by {generated_by or 'PulmoAI user'}", st["sub"]))
    story.append(Spacer(1, 8))

    flt = [
        ("Date range", _date_range(filters.get("start_date"), filters.get("end_date"))),
        ("Disease", str(filters.get("disease") or "all").replace("all", "All")),
        ("Result", str(filters.get("status") or "all").replace("all", "All")),
    ]
    if filters.get("patient"):
        flt.append(("Patient", filters["patient"]))
    if filters.get("doctor"):
        flt.append(("Doctor", filters["doctor"]))
    if filters.get("search"):
        flt.append(("Search", filters["search"]))
    if filters.get("source") in ("single", "full_screening"):
        flt.append(("Source", "Single analysis" if filters["source"] == "single" else "Full screening"))
    story.append(_kv_table(flt, st))

    # counts per disease
    total = len(rows)
    positives = sum(r["positive"] for r in rows)
    story.append(Paragraph("Summary", st["h2"]))
    n_patients = len({r.get("patient_code") or r["patient_id"] for r in rows})
    n_doctors = len({r["doctor_id"] for r in rows if r.get("doctor_id")})
    story.append(Paragraph(f"{total} scans - {n_patients} patient{'s' if n_patients != 1 else ''} - "
                           f"{n_doctors} doctor{'s' if n_doctors != 1 else ''}", st["body"]))
    story.append(Spacer(1, 6))
    counts = []
    for _, (_, disease, scan) in COMPONENTS.items():
        d_rows = [r for r in rows if r["disease"] == disease]
        if d_rows:
            p = sum(r["positive"] for r in d_rows)
            pats = len({r.get("patient_code") or r["patient_id"] for r in d_rows})
            counts.append([disease, scan, str(pats), str(len(d_rows)), str(p), str(len(d_rows) - p)])
    head = ["Disease", "Scan", "Patients", "Scans", "Positive", "Normal"]
    table = [head] + counts + [["Total", "", str(n_patients), str(total), str(positives), str(total - positives)]]
    t = Table(table, colWidths=[45 * mm, 25 * mm, 27 * mm, 27 * mm, 28 * mm, 28 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # result list
    story.append(Paragraph(f"Results ({total})", st["h2"]))
    if not rows:
        story.append(Paragraph("No results match these filters.", st["body"]))
    else:
        data = [["Date", "Patient", "Disease", "Result", "Conf.", "Urgency", "Doctor"]]
        for r in rows:
            code, pname = r.get("patient_code") or r["patient_id"], r.get("patient_name", "")
            patient = f"{code}<br/><font color='#566874' size='7.5'>{pname}</font>" if pname else code
            colour = "#B42318" if r["positive"] else "#15803D"
            conf = fmt_conf(r["confidence"])
            data.append([
                Paragraph(fmt_dt(r["created_at"]), st["cell"]),
                Paragraph(patient, st["cell"]),
                Paragraph(r["disease"], st["cell"]),
                Paragraph(f"<font color='{colour}'><b>{r['prediction']}</b></font>", st["cell"]),
                Paragraph(conf, st["cellr"]),
                Paragraph(str(r["urgency"] or "-") if r["positive"] else "-", st["cell"]),
                Paragraph(r.get("doctor_name") or "-", st["cell"]),
            ])
        t = Table(data, colWidths=[27 * mm, 27 * mm, 25 * mm, 37 * mm, 15 * mm, 19 * mm, 30 * mm], repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
        ]
        for i in range(2, len(data), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
        t.setStyle(TableStyle(style))
        story.append(t)
        if len(rows) >= MAX_ROWS:
            story.append(Paragraph(f"Only the newest {MAX_ROWS} results are shown. Narrow the filters to see others.",
                                   st["small"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(DISCLAIMER, st["small"]))
    short = title.split(" - ")[0]
    doc.build(story, onFirstPage=_header_footer(short), onLaterPages=_header_footer(short))
    return buf.getvalue()

# ── 2. single result report ──────────────────────────────────────
def _patient_section(result: dict, patient: Optional[dict], st) -> list:
    p = patient or {}
    pairs = [("Patient ID", p.get("patient_id") or result.get("patient_id", "-")),
             ("Name", p.get("full_name") or p.get("name") or "-")]
    for key, label in (("age", "Age"), ("gender", "Gender"), ("sex", "Sex"), ("contact", "Contact"),
                       ("phone", "Phone")):
        if p.get(key) not in (None, ""):
            pairs.append((label, p[key]))
    return [Paragraph("Patient", st["h2"]), _kv_table(pairs, st)]


def _result_box(result: dict, st) -> Table:
    positive = is_positive(result.get("prediction"))
    colour = POS if positive else NEG
    conf = result.get("confidence")
    conf_txt = f"{fmt_conf(conf)} confidence" if isinstance(conf, (int, float)) else ""
    urgency = result.get("urgency") or result.get("severity")
    box = Table([
        [Paragraph(f"<font color='{colour.hexval()}'>{pretty_prediction(result.get('prediction'))}</font>", st["big"])],
        [Paragraph(" - ".join(x for x in (conf_txt, f"Urgency: {urgency}" if urgency and positive else "") if x),
                   st["body"])],
    ], colWidths=[180 * mm])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colour),
        ("LINEBEFORE", (0, 0), (0, -1), 4, colour),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF3F2") if positive else colors.HexColor("#F0FAF3")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return box


def _details_table(result: dict, scan: str, doctor_name: str, st) -> Table:
    """Every other stored field (component-specific indicators)."""
    positive = is_positive(result.get("prediction"))
    details = [("Scan type", scan), ("Analysed at", fmt_dt(result.get("created_at"))),
               ("Requested by", doctor_name or "-")]
    hidden = SKIP_FIELDS | HIDE_IN_PDF | {"patient_id", "doctor_id", "prediction", "confidence", "created_at",
                                          "component", "status", "is_xray", "result_id"}
    if not positive:
        hidden |= {"urgency", "severity"}
    for k, v in result.items():
        if k in hidden or v is None or (isinstance(v, str) and len(v) > 300):
            continue
        if k in HIDE_WHEN_EMPTY and not v:
            continue
        details.append((FIELD_LABELS.get(k, k.replace("_", " ").capitalize()), fmt_field(k, v)))
    return _kv_table(details, st)


def _clinician_and_disclaimer(st) -> list:
    return [
        KeepTogether([
            Paragraph("Clinician review", st["h2"]),
            _kv_table([("Findings confirmed", "[ ] Yes    [ ] No"), ("Comments", "<br/><br/><br/>"),
                       ("Name / signature", "<br/><br/>"), ("Date", "")], st),
        ]),
        Spacer(1, 10),
        Paragraph(DISCLAIMER, st["small"]),
    ]


def build_result_pdf(result: dict, component: str, patient: Optional[dict], doctor_name: str,
                     generated_by: str) -> bytes:
    coll = COMPONENT_TO_COLLECTION[component]
    _, disease, scan = COMPONENTS[coll]
    st = _styles()
    buf = BytesIO()
    doc = _doc(buf, f"PulmoAI {disease} report")
    story = []

    story.append(Paragraph(f"{disease} Analysis Report", st["title"]))
    story.append(Paragraph(f"Report ID {result.get('_id')} - Generated {now_local()} by "
                           f"{generated_by or 'PulmoAI user'}", st["sub"]))
    story += _patient_section(result, patient, st)
    story.append(Paragraph("AI result", st["h2"]))
    story.append(_result_box(result, st))
    story.append(Paragraph("Details", st["h2"]))
    story.append(_details_table(result, scan, doctor_name, st))
    story += _clinician_and_disclaimer(st)

    title = f"{disease} Report"
    doc.build(story, onFirstPage=_header_footer(title), onLaterPages=_header_footer(title))
    return buf.getvalue()


# ── 3. full screening report (one X-ray, several diseases) ───────
def build_screening_pdf(items: list, patient: Optional[dict], doctor_name: str, generated_by: str) -> bytes:
    """items = [(component, result_doc), ...] in the order they should appear."""
    st = _styles()
    buf = BytesIO()
    doc = _doc(buf, "PulmoAI full screening report")
    story = []
    first = items[0][1]

    story.append(Paragraph("Full Screening Report", st["title"]))
    story.append(Paragraph(f"Generated {now_local()} by {generated_by or 'PulmoAI user'}", st["sub"]))
    story += _patient_section(first, patient, st)

    # overview table: one row per disease
    story.append(Paragraph("Screening overview", st["h2"]))
    data = [["Disease", "Result", "Confidence", "Urgency"]]
    for component, r in items:
        _, disease, _ = COMPONENTS[COMPONENT_TO_COLLECTION[component]]
        positive = is_positive(r.get("prediction"))
        colour = "#B42318" if positive else "#15803D"
        conf = r.get("confidence")
        data.append([
            Paragraph(disease, st["cell"]),
            Paragraph(f"<font color='{colour}'><b>{pretty_prediction(r.get('prediction'))}</b></font>", st["cell"]),
            Paragraph(fmt_conf(conf), st["cell"]),
            Paragraph(str((r.get("urgency") or r.get("severity") or "-") if positive else "-"), st["cell"]),
        ])
    t = Table(data, colWidths=[45 * mm, 70 * mm, 30 * mm, 35 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # one section per disease
    for component, r in items:
        _, disease, scan = COMPONENTS[COMPONENT_TO_COLLECTION[component]]
        story.append(KeepTogether([
            Paragraph(disease, st["h2"]),
            _result_box(r, st),
            Spacer(1, 6),
            _details_table(r, scan, doctor_name, st),
        ]))

    story += _clinician_and_disclaimer(st)
    doc.build(story, onFirstPage=_header_footer("Full Screening Report"),
              onLaterPages=_header_footer("Full Screening Report"))
    return buf.getvalue()