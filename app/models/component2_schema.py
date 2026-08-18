from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class PneumoniaMongoResult(BaseModel):
    """
    MongoDB document model for Component 2 (Pneumonia Detection & Severity).
    Matches the unified CDSS schema format while preserving diagnosis, confidence, 
    severity, explanation_image, and timestamp.
    """
    status: str = Field("success", description="Status of the analysis")
    is_xray: bool = Field(True, description="Whether the uploaded image is a valid chest X-ray")
    patient_id: str = Field(..., description="The associated patient ID")
    doctor_id: Optional[str] = Field(None, description="The ID of the diagnosing doctor")
    component: str = Field("pneumonia", description="Component identifier")
    prediction: str = Field(..., description="Diagnosis result (e.g. 'Pneumonia Detected' or 'Normal')")
    diagnosis: Optional[str] = Field(None, description="Alias for prediction diagnosis label")
    confidence: float = Field(..., description="Confidence score percentage (0-100)")
    raw_score: float = Field(..., description="Raw model output probability (0.0 to 1.0)")
    severity: str = Field(..., description="Calculated severity (e.g. Mild, Moderate, Severe, Normal)")
    affected_area_percent: Optional[float] = Field(None, description="Quantified affected lung area percentage from Grad-CAM")
    mean_intensity: Optional[float] = Field(None, description="Mean intensity of activation in affected areas")
    explanation_image: Optional[str] = Field(None, description="Base64 encoded Grad-CAM heatmap visualization")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timezone-aware UTC timestamp")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO formatted creation timestamp string")

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class PneumoniaPredictionResponse(BaseModel):
    """
    API Response model for Component 2 Pneumonia prediction endpoint.
    """
    database_record_id: Optional[str] = Field(None, description="The MongoDB document ID")
    patient_id: str = Field(..., description="The associated patient ID")
    filename: Optional[str] = Field(None, description="The original uploaded filename")
    diagnosis: str = Field(..., description="Diagnosis label")
    confidence: str = Field(..., description="Formatted confidence percentage string")
    severity: str = Field(..., description="Severity classification")
    affected_area_percent: Optional[float] = Field(None, description="Grad-CAM affected lung area percentage")
    mean_intensity: Optional[float] = Field(None, description="Grad-CAM mean activation intensity")
    explanation_image: Optional[str] = Field(None, description="Base64 Grad-CAM heatmap image")
