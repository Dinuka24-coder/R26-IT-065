# Unified Clinical Decision Support System for Pulmonary Diseases
### Backend API — Python FastAPI + MongoDB

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The 4 Components](#the-4-components)
3. [Tech Stack](#tech-stack)
4. [Folder Structure](#folder-structure)
5. [Folder Explanation](#folder-explanation)
6. [How a Request Flows (Component 1 Example)](#how-a-request-flows)
7. [Getting Started](#getting-started)
8. [API Endpoints](#api-endpoints)
9. [Environment Variables](#environment-variables)
10. [Docker Setup](#docker-setup)

---

## Project Overview

Diagnosing pulmonary diseases today is a fragmented process. A patient goes through multiple hand-offs between GPs, radiographers, radiologists, and pulmonologists — each running isolated tests in separate systems.

This project builds a **Single Playground**: a unified, web-based Clinical Decision Support System (CDSS) where a single attending doctor can evaluate a patient's full pulmonary profile from one dashboard — no waiting on departmental hand-offs.

The backend is a **FastAPI** application exposing REST endpoints for four AI-powered diagnostic components, backed by **MongoDB** for flexible medical data storage.

---

## The 4 Components

| # | Component | Input | AI Model | Output |
|---|-----------|-------|----------|--------|
| 1 | **Pneumothorax Detection** | Chest X-ray | MobileNet / EfficientNet + Grad-CAM | Prediction, confidence score, heatmap overlay, urgency level (High / Moderate / Low) |
| 2 | **Pneumonia Diagnosis & Severity** | Chest X-ray | MobileNetV2 + Weakly Supervised Localization | Disease label, severity percentage, lung damage boundary overlay |
| 3 | **Tuberculosis Detection** | Chest X-ray | GhostNet + MobileViT + Two-stage XAI | TB lesion segmentation (cavities, nodules, consolidations), pixel-level mask, lung impact score |
| 4 | **3D Lung Cancer Detection** | CT Scan (DICOM) | 3D Volumetric CNN | Nodule / tumor localization in 3D, detects lesions hidden behind ribs or heart tissue |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | Python FastAPI |
| Database | MongoDB (via Motor async driver) |
| File Storage | MongoDB GridFS (for X-rays, CT scans, heatmaps) |
| ML Framework | TensorFlow / PyTorch |
| Containerization | Docker + Docker Compose |
| Image Processing | OpenCV, Pillow |
| DICOM Handling | PyDICOM (Component 4) |

**Why MongoDB over MySQL?**
Medical imaging results from each component have different structures (heatmaps, bounding boxes, severity scores, 3D nodule coordinates). MongoDB's flexible document model handles this naturally, and GridFS provides built-in large file storage for scans alongside their metadata — no painful schema migrations needed.

---

## Folder Structure

```
pulmonary-cdss-backend/
│
├── app/
│   ├── main.py                          # FastAPI app entry point
│   ├── config.py                        # Settings & environment variables
│   ├── database.py                      # MongoDB connection
│   │
│   ├── api/                             # HTTP route handlers (front door)
│   │   └── v1/
│   │       ├── router.py
│   │       ├── patients.py
│   │       ├── component1_pneumothorax.py
│   │       ├── component2_pneumonia.py
│   │       ├── component3_tuberculosis.py
│   │       └── component4_lung_cancer.py
│   │
│   ├── models/                          # Pydantic request/response schemas
│   │   ├── patient.py
│   │   ├── scan.py
│   │   ├── component1_schema.py
│   │   ├── component2_schema.py
│   │   ├── component3_schema.py
│   │   └── component4_schema.py
│   │
│   ├── services/                        # Business logic coordinator
│   │   ├── patient_service.py
│   │   ├── component1_service.py
│   │   ├── component2_service.py
│   │   ├── component3_service.py
│   │   └── component4_service.py
│   │
│   ├── ml/                              # All AI / deep learning code
│   │   ├── component1/                  # Pneumothorax
│   │   │   ├── model.py
│   │   │   ├── inference.py
│   │   │   ├── gradcam.py
│   │   │   ├── urgency.py
│   │   │   └── weights/
│   │   │       └── pneumothorax_model.h5
│   │   ├── component2/                  # Pneumonia
│   │   │   ├── model.py
│   │   │   ├── inference.py
│   │   │   ├── gradcam.py
│   │   │   ├── severity.py
│   │   │   └── weights/
│   │   │       └── pneumonia_model.h5
│   │   ├── component3/                  # Tuberculosis
│   │   │   ├── model.py
│   │   │   ├── inference.py
│   │   │   ├── segmentation.py
│   │   │   ├── xai.py
│   │   │   └── weights/
│   │   │       └── tb_model.h5
│   │   └── component4/                  # Lung Cancer (CT)
│   │       ├── model.py
│   │       ├── inference.py
│   │       ├── nodule_detector.py
│   │       ├── ct_preprocessor.py
│   │       └── weights/
│   │           └── lung_cancer_3d.pt
│   │
│   ├── repositories/                    # MongoDB data access layer
│   │   ├── patient_repo.py
│   │   ├── scan_repo.py
│   │   └── result_repo.py
│   │
│   ├── utils/                           # Shared helper tools
│   │   ├── image_utils.py
│   │   ├── dicom_utils.py
│   │   ├── file_storage.py
│   │   ├── heatmap_utils.py
│   │   └── response_utils.py
│   │
│   └── middleware/
│       ├── auth.py
│       └── logging.py
│
├── tests/
│   ├── test_component1.py
│   ├── test_component2.py
│   ├── test_component3.py
│   ├── test_component4.py
│   └── conftest.py
│
├── static/outputs/                      # Temporary heatmap/overlay images
├── .env
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Folder Explanation

### `app/main.py` — Entry Point
The starting file of the entire application. It creates the FastAPI app, registers all the routers, and connects to MongoDB on startup.

---

### `app/api/` — Front Door (Route Handlers)
Each file here maps to one component's HTTP endpoints. This layer only receives requests and sends responses — it does **no AI work** itself.

```
POST /api/v1/pneumothorax/predict     → component1_pneumothorax.py
POST /api/v1/pneumonia/predict        → component2_pneumonia.py
POST /api/v1/tuberculosis/predict     → component3_tuberculosis.py
POST /api/v1/lung-cancer/predict      → component4_lung_cancer.py
```

---

### `app/models/` — Data Shapes
Pydantic schema files that define exactly what the input and output of each API call looks like. Think of these as the official form templates — they validate data automatically.

```python
# Example: component1_schema.py
class PneumothoraxResult(BaseModel):
    prediction: str        # "Pneumothorax Detected"
    confidence: float      # 0.94
    urgency_level: str     # "High"
    heatmap_url: str       # URL to the overlay image
```

---

### `app/services/` — The Coordinator
Each service file orchestrates the full process for one component. It is the only file that talks to both the ML layer and the database layer.

```
Receives image from API
    → Preprocesses it (via utils/)
    → Sends to ML model for prediction
    → Saves result to MongoDB (via repositories/)
    → Returns final response to API
```

---

### `app/ml/` — The AI Brain
All deep learning code lives here, organized per component. Each component folder has the same internal structure:

| File | Job |
|------|-----|
| `model.py` | Loads the trained model weights from disk once at startup |
| `inference.py` | Runs the image through the model, returns prediction + confidence |
| `gradcam.py` | Generates the Grad-CAM heatmap showing *where* disease is detected |
| `urgency.py` *(C1)* | Classifies urgency as High / Moderate / Low |
| `severity.py` *(C2)* | Calculates the percentage of lung compromised |
| `segmentation.py` *(C3)* | Produces pixel-level TB lesion masks |
| `nodule_detector.py` *(C4)* | Localizes 3D nodules/tumors in CT volume |
| `ct_preprocessor.py` *(C4)* | Converts DICOM files to normalized 3D arrays |
| `weights/` | Stores the trained `.h5` / `.pt` model files |

---

### `app/repositories/` — The Filing Cabinet
MongoDB query code lives here. These files only know how to read and write data. They have no knowledge of AI, HTTP, or business logic.

```python
# result_repo.py
async def save(result):
    await db["results"].insert_one(result.dict())

async def get_by_patient_id(patient_id):
    return await db["results"].find({"patient_id": patient_id})
```

---

### `app/utils/` — Shared Tools
Helper functions used across all 4 components:

| File | Purpose |
|------|---------|
| `image_utils.py` | Resize X-rays, normalize pixel values for model input |
| `heatmap_utils.py` | Alpha-blend Grad-CAM overlay onto the original X-ray |
| `file_storage.py` | Upload/download scan images and heatmaps via MongoDB GridFS |
| `dicom_utils.py` | Parse DICOM CT files into numpy arrays (Component 4 only) |
| `response_utils.py` | Wrap all API responses in a consistent JSON envelope |

---

## How a Request Flows

Using **Component 1 — Pneumothorax Detection** as an example:

```
Doctor uploads chest X-ray via frontend
             ↓
[api/component1_pneumothorax.py]     Receives the HTTP POST request
             ↓
[services/component1_service.py]     Coordinates the full pipeline
             ↓
[utils/image_utils.py]               Resizes to 224×224, normalizes pixels
             ↓
[ml/component1/inference.py]         MobileNet predicts: "Pneumothorax Detected" (94%)
             ↓
[ml/component1/gradcam.py]           Generates heatmap highlighting affected region
             ↓
[ml/component1/urgency.py]           Assigns urgency: "HIGH"
             ↓
[utils/heatmap_utils.py]             Overlays heatmap on original X-ray image
             ↓
[utils/file_storage.py]              Saves overlay image to MongoDB GridFS
             ↓
[repositories/result_repo.py]        Saves result document to MongoDB
             ↓
Doctor receives:  prediction + confidence + heatmap image + urgency level
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- MongoDB 7+
- Docker & Docker Compose (recommended)
- NVIDIA GPU (recommended for Component 4 CT inference)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/pulmonary-cdss-backend.git
cd pulmonary-cdss-backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment file and fill in values
cp .env.example .env

# 5. Place trained model weights in their folders
#    app/ml/component1/weights/pneumothorax_model.h5
#    app/ml/component2/weights/pneumonia_model.h5
#    app/ml/component3/weights/tb_model.h5
#    app/ml/component4/weights/lung_cancer_3d.pt

# 6. Run the server
uvicorn app.main:app --reload --port 8000
```

### Run with Docker (Recommended)

```bash
docker-compose up --build
```

This spins up:
- `fastapi-app` on port `8000`
- `mongodb` on port `27017`

---

## API Endpoints

### Patient Management
```
POST   /api/v1/patients/          Create new patient record
GET    /api/v1/patients/{id}      Get patient details and history
```

### Component 1 — Pneumothorax
```
POST   /api/v1/pneumothorax/predict
Body:  { file: <chest_xray_image> }

Response:
{
  "prediction": "Pneumothorax Detected",
  "confidence": 0.94,
  "urgency_level": "High",
  "heatmap_url": "/static/outputs/heatmap_abc123.png"
}
```

### Component 2 — Pneumonia
```
POST   /api/v1/pneumonia/predict
Body:  { file: <chest_xray_image> }

Response:
{
  "prediction": "Pneumonia",
  "confidence": 0.89,
  "severity_percentage": 34.7,
  "overlay_url": "/static/outputs/overlay_xyz456.png"
}
```

### Component 3 — Tuberculosis
```
POST   /api/v1/tuberculosis/predict
Body:  { file: <chest_xray_image> }

Response:
{
  "prediction": "Tuberculosis Detected",
  "lesion_types": ["cavity", "nodule"],
  "lung_impact_score": 0.61,
  "mask_url": "/static/outputs/mask_tb789.png"
}
```

### Component 4 — Lung Cancer (CT)
```
POST   /api/v1/lung-cancer/predict
Body:  { file: <ct_dicom_file> }

Response:
{
  "prediction": "Malignant Nodule Detected",
  "nodule_count": 2,
  "nodule_locations": [{"x": 112, "y": 88, "z": 34, "diameter_mm": 8.2}],
  "confidence": 0.91
}
```

Full interactive API docs available at: `http://localhost:8000/docs`

---

## Environment Variables

```env
# .env.example

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=pulmonary_cdss

# App
APP_ENV=development
SECRET_KEY=your_secret_key_here
API_PREFIX=/api/v1

# Model paths (override if needed)
C1_MODEL_PATH=app/ml/component1/weights/pneumothorax_model.h5
C2_MODEL_PATH=app/ml/component2/weights/pneumonia_model.h5
C3_MODEL_PATH=app/ml/component3/weights/tb_model.h5
C4_MODEL_PATH=app/ml/component4/weights/lung_cancer_3d.pt
```

---

## Docker Setup

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - mongo
    volumes:
      - ./static:/app/static

  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

---

## Why One File, One Job?

This project follows the **Single Responsibility Principle**:

- If Grad-CAM logic breaks → only touch `gradcam.py`
- If you swap MobileNet for EfficientNet → only change `model.py`
- If MongoDB moves to PostgreSQL → only rewrite `repositories/`
- If severity calculation changes → only edit `severity.py`

Each of the 4 components follows the **exact same folder pattern** inside `ml/`, making it easy for different team members to work on different components independently.

---

## Project Team

> Unified Clinical Decision Support System for Pulmonary Diseases  
> Research Project — Department of [Your Department]  
> [Your University / Institution]