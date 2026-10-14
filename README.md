# PulmoAI — Multi-Modal Pulmonary Disease Detection System
### Backend API — Python FastAPI + MongoDB

> SLIIT Research Project **R26-IT-065** — Unified Clinical Decision Support System (CDSS) for Pulmonary Diseases

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The 4 Components](#the-4-components)
3. [Main Features](#main-features)
4. [Tech Stack](#tech-stack)
5. [Folder Structure](#folder-structure)
6. [How a Request Flows](#how-a-request-flows)
7. [Getting Started](#getting-started)
8. [Environment Variables](#environment-variables)
9. [API Endpoints](#api-endpoints)
10. [Shared Result Format](#shared-result-format)
11. [PDF Reports](#pdf-reports)
12. [MongoDB Collections](#mongodb-collections)
13. [Troubleshooting](#troubleshooting)
14. [Project Team](#project-team)

---

## Project Overview

Diagnosing pulmonary diseases is often fragmented: a patient moves between GPs,
radiographers, radiologists and pulmonologists, and each test lives in a separate system.

PulmoAI is a **single web-based Clinical Decision Support System** where a doctor can
check a patient for several lung diseases from one dashboard, see an explainable AI
result (heatmaps), and download a PDF report.

The backend is a **FastAPI** application that exposes REST endpoints for four AI
components, a Full Screening pipeline, dashboards, history and PDF reports,
with **MongoDB** for storage.

> AI results are decision support only and must be reviewed and confirmed by a qualified clinician.

---

## The 4 Components

| # | Component | Input | Output |
|---|-----------|-------|--------|
| 1 | **Pneumothorax Detection** | Chest X-ray | Prediction, confidence, Boundary-Aware Grad-CAM heatmap, attention indicators, urgency (High / Moderate / Low) |
| 2 | **Pneumonia Diagnosis & Severity** | Chest X-ray | Prediction, confidence, severity, affected area %, Grad-CAM explanation |
| 3 | **Tuberculosis Detection** | Chest X-ray | Healthy / Other Abnormality / Tuberculosis, confidence, clinical note, heatmap. Includes a gatekeeper that rejects non-chest-X-ray images |
| 4 | **Lung Cancer Classification** | CT scan image | Adenocarcinoma / Large Cell / Squamous Cell Carcinoma / Normal, class probabilities, input check, Grad-CAM (two branches) |

**Full Screening** runs one chest X-ray through components 1–3 in parallel and
returns one combined result.

---

## Main Features

- **4 AI components** with explainable heatmaps
- **Full Screening** – one X-ray, all X-ray engines in parallel, one combined verdict
- **Input validation** – a central chest X-ray validator plus each engine's own check
- **PDF reports** – single result, Full Screening, summary, and Doctor / Disease / Patient reports
- **Reports list** with search and filters (date, disease, result, source, doctor, patient)
- **Dashboard** – scans, analyses, positive cases, weekly chart, disease distribution, recent activity
- **Role-based access** – Admin sees everything, Doctors see only their own results
- **Sri Lanka time (UTC+05:30)** in all reports and dashboard counts

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | Python FastAPI |
| Database | MongoDB Atlas (Motor async driver) |
| ML Framework | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow, NumPy |
| PDF Reports | ReportLab |
| Auth | JWT |

**Why MongoDB?** Each component stores a different result shape (heatmaps, bounding
boxes, class probabilities, severity scores). MongoDB's flexible documents handle this
without schema migrations.

---

## Folder Structure

```
R26-IT-065/
├── app/
│   ├── main.py                     # App entry point, CORS, log setup, router
│   ├── config.py                   # Settings from .env
│   ├── database.py                 # MongoDB connection
│   │
│   ├── api/v1/                     # HTTP routes (no AI logic here)
│   │   ├── router.py               # Registers all routers
│   │   ├── component2_pneumonia.py
│   │   ├── component3_tuberculosis.py
│   │   ├── screening.py            # Full Screening
│   │   ├── dashboard.py            # Dashboard stats and charts
│   │   ├── reports.py              # Prediction history
│   │   ├── report_pdf.py           # Reports list + PDF downloads
│   │   └── ...                     # auth, users, patients, other components
│   │
│   ├── services/                   # Business logic for each feature
│   │   ├── comp1_service.py        # Pneumothorax
│   │   ├── comp2_service.py        # Pneumonia
│   │   ├── comp3_service.py        # Tuberculosis
│   │   ├── screening_service.py    # Full Screening pipeline
│   │   └── report_service.py       # Report data + PDF building
│   │
│   ├── ml_models/                  # AI code and model weights, one folder per component
│   │   ├── component1/             # Pneumothorax (model, Grad-CAM, urgency, X-ray validator)
│   │   ├── component2/             # Pneumonia
│   │   ├── component3/             # Tuberculosis (controller, gatekeeper cascade)
│   │   └── component4/             # Lung cancer
│   │
│   ├── models/                     # Pydantic request/response schemas
│   ├── repositories/               # MongoDB read/write helpers (e.g. result_repo.py)
│   ├── core/                       # Auth dependencies (current user, roles)
│   └── utils/                      # Shared helpers (collection cache, image checks)
│
├── requirements.txt
├── .env.example
└── README.md
```

**One file, one job:** routes only handle HTTP, services coordinate the work,
`ml_models/` only does AI, and `repositories/` only talks to MongoDB.
Each team member works inside their own component folder.

---

## How a Request Flows

**Single analysis (Component 1 example)**

```
Doctor uploads a chest X-ray
        ↓
api/v1 (pneumothorax route)        receives the request
        ↓
services/comp1_service.py          coordinates the pipeline
        ↓
ml_models/component1               validate X-ray → predict → Grad-CAM → urgency
        ↓
MongoDB pneumothorax_results       result saved (with doctor_id, created_at)
        ↓
Response: prediction, confidence, heatmap, urgency, result_id
```

**Full Screening**

```
POST /screening/full
        ↓
Central chest X-ray validator      rejects non-chest-X-ray images
        ↓
Pneumothorax ┐
Pneumonia    ├─ run in parallel, each with its own check, each saves its own result
Tuberculosis ┘
        ↓
screening_results                  one summary record (verdict, overall urgency, result_ids)
        ↓
Each engine result is tagged with source="full_screening" and screening_id
```

---

## Getting Started

### Prerequisites
- Python 3.10+ (developed on Python 3.13, Windows)
- A MongoDB database (MongoDB Atlas or local)
- Trained model weights in each `app/ml_models/componentX/weights/` folder

### Local setup (Windows / PowerShell)

```powershell
# 1. Clone and enter the project
git clone <repo-url>
cd R26-IT-065

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file and fill in the values
copy .env.example .env

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

On macOS / Linux use `source .venv/bin/activate` and `cp .env.example .env`.

The server is ready when you see:

```
✅ Connected to MongoDB: pulmonary_cdss
📋 Collections cached: [...]
INFO:     Application startup complete.
```

Interactive API docs: **http://127.0.0.1:8000/docs**

### Updating requirements.txt

```powershell
pip freeze > requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values. Main settings:

| Setting | Purpose |
|---|---|
| MongoDB connection string | Atlas or local MongoDB URL |
| Database name | `pulmonary_cdss` |
| JWT secret key | Used to sign login tokens |
| `OPENAI_API_KEY` *(optional)* | Used by the TB gatekeeper. If not set, it uses the CNN + heuristic check only |

> Never commit `.env` to Git.

---

## API Endpoints

All endpoints start with `/api/v1`. Most need a JWT: `Authorization: Bearer <token>`.
See `/docs` for the full list and request bodies.

### Auth, users and patients
| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/login` | Log in (form: username, password) |
| GET | `/users/me` | Current user |
| GET / POST | `/users/doctors` | List / add doctors (admin) |
| PUT / DELETE | `/users/doctors/{id}` | Edit / remove a doctor (admin) |
| GET / POST | `/patients` | List / register patients |
| GET | `/patients/search?q=` | Search patients |
| GET / PUT / DELETE | `/patients/{id}` | Patient details |
| POST | `/patients/{id}/notes` | Add a clinical note |

### AI components
| Method | Path | Purpose |
|---|---|---|
| POST | `/pneumothorax/predict` | Component 1 (form: patient_id, file) |
| POST | `[pneumonia predict path]` | Component 2 |
| POST | `/tuberculosis/predict` | Component 3 |
| POST | `/tuberculosis/gatekeeper/predict` | TB gatekeeper only (testing, nothing saved) |
| POST | `[lung cancer predict path]` | Component 4 |
| POST | `/screening/full` | Full Screening (form: patient_id, file) |

Every successful prediction response includes **`result_id`** (the saved record ID),
which the frontend uses for the PDF download button.

### Dashboard and history
| Method | Path | Purpose |
|---|---|---|
| GET | `/dashboard/overview` | Stats, weekly chart, disease distribution and recent activity in one call |
| GET | `/dashboard/stats` | Stat cards only |
| GET | `/dashboard/weekly-volume` | Scans per day (last 7 days) |
| GET | `/dashboard/disease-distribution` | Results per disease |
| GET | `/reports/history?patient_id=&doctor_id=` | History for Prediction History, Patient and Doctor pages |

**Dashboard counts**
- `total_scans` – images uploaded (one Full Screening = **1** scan)
- `total_analyses` – model results (one Full Screening = **3** analyses)
- `total_screenings`, `single_analyses`, `today_scans`, `today_predictions`

### Reports and PDFs
| Method | Path | Purpose |
|---|---|---|
| GET | `/reports/pdf/list` | Results for the Reports page (JSON) |
| GET | `/reports/pdf/summary` | Summary PDF; becomes a Doctor, Disease or Patient report when that filter is set |
| GET | `/reports/pdf/result/{component}/{result_id}` | PDF for one result |
| GET | `/reports/pdf/screening?pneumothorax=&pneumonia=&tuberculosis=` | One combined Full Screening PDF |

**Filters** for `list` and `summary`: `start_date`, `end_date` (YYYY-MM-DD),
`disease`, `status` (all / Positive / Normal), `source` (all / single / full_screening),
`patient_id`, `doctor_id` (admin only), `search`.

`component` is one of: `pneumothorax`, `pneumonia`, `tuberculosis`, `lungcancer`.
Add `inline=true` to open the PDF in the browser instead of downloading it.

---

## Shared Result Format

All components save results in the same basic format so that the Dashboard,
History, Reports and PDFs work for everyone:

```python
{
    "patient_id": "...",                      # same patient ID as other components
    "doctor_id":  str(user["_id"]),           # who requested the analysis
    "component":  "pneumonia",                # pneumothorax | pneumonia | tuberculosis | lungcancer
    "prediction": "Pneumonia Detected",       # positive results contain "Detected"
    "confidence": 84.55,                      # 0–100, confidence in the predicted class
    "urgency":    "High",                     # High | Moderate | Low
    "created_at": datetime.now(timezone.utc).isoformat(),
    # ...plus any component-specific fields
}
```

Full Screening results also have `source: "full_screening"` and `screening_id`.

**Tips for component owners**
- Give fields clear names. The PDF shows every saved field, e.g. `affected_area_percent`.
- Large fields (heatmaps, images) are skipped in the PDF and lists automatically.
- Return `result_id` in the API response.

---

## PDF Reports

Built with ReportLab in `app/services/report_service.py`.

| Report | Contents |
|---|---|
| **Single result** | Patient details, AI result, confidence, urgency, model details, clinician review section |
| **Full Screening** | Patient details, overview table, one section per engine |
| **Summary** | Filters, counts per disease (patients, scans, positive, normal), full result list |
| **Doctor / Disease / Patient** | Summary report filtered for one doctor, disease or patient |

All times are shown in **Sri Lanka time (UTC+05:30)**.

> `report_service.py` is shared by all components. Pull the latest version before
> changing it so other people's fixes are not lost.

---

## MongoDB Collections

| Collection | Contents |
|---|---|
| `users` | Admins and doctors |
| `patients` | Patient records |
| `pneumothorax_results` | Component 1 results |
| `pneumonia_results` | Component 2 results |
| `tuberculosis_results` | Component 3 results |
| `lung_cancer_results` | Component 4 results |
| `screening_results` | Full Screening summaries (with each engine's `result_id`) |

> The server caches the list of collections at startup. If a collection is created
> while the server is running (first result of a new component), **restart the
> backend** so it shows on the Dashboard and Reports.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `NetworkTimeout` / "Database is not responding" | Check your internet. In MongoDB Atlas → **Network Access**, make sure your current IP is allowed |
| A component's results do not show on the Dashboard | Restart the backend (collection cache) |
| Results missing for a doctor on the Reports page | The result was saved without `doctor_id` |
| PDF download button does not appear | The prediction response has no `result_id` |
| TensorFlow GPU / Keras optimizer warnings | Harmless on Windows; they are hidden at the top of `main.py` |
| CORS error in the browser | `main.py` must allow `http://localhost:5173`, and use only **one** CORS middleware |

---

## Project Team

**PulmoAI — Unified Clinical Decision Support System for Pulmonary Diseases**
SLIIT Research Project — R26-IT-065

| Component | Member              |
|---|---------------------|
| 1 – Pneumothorax Detection | Navodya Dhanushka   |
| 2 – Pneumonia Diagnosis & Severity | Dinuka Amarasekara  |
| 3 – Tuberculosis Detection | Chanuka Nanayakkara |
| 4 – Lung Cancer Classification | Anuththara Bandara  |