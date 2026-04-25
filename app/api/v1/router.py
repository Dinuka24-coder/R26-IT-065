from fastapi import APIRouter
from app.api.v1 import (
    patients,
    component1_pneumothorax,
    component2_pneumonia,
    component3_tuberculosis,
    component4_lung_cancer
)

router = APIRouter()

router.include_router(patients.router,                  prefix="/patients",     tags=["Patients"])
router.include_router(component1_pneumothorax.router,   prefix="/pneumothorax", tags=["Component 1 - Pneumothorax"])
router.include_router(component2_pneumonia.router,      prefix="/pneumonia",    tags=["Component 2 - Pneumonia"])
router.include_router(component3_tuberculosis.router,   prefix="/tuberculosis", tags=["Component 3 - Tuberculosis"])
router.include_router(component4_lung_cancer.router,    prefix="/lung-cancer",  tags=["Component 4 - Lung Cancer"])