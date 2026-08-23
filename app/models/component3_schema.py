from typing import List, Optional
from pydantic import BaseModel, Field
 
 
class TBPredictionResponse(BaseModel):
    database_record_id: Optional[str] = Field(None, description="The MongoDB document ID.")
    patient_id: Optional[str] = Field(None, description="The associated patient ID.")
    filename: Optional[str] = Field(None, description="The original uploaded filename.")
    status: str = Field(..., description="'success' if the scan was processed.")
    diagnosis: str = Field(..., description="Healthy, Non-TB, or Tuberculosis.")
    confidence_score: float = Field(..., description="Confidence percentage out of 100.")
    message: Optional[str] = Field(None, description="Human-readable status message.")
    bounding_box: Optional[List[float]] = Field(None, description="TB localization box [x_min, y_min, x_max, y_max] (normalized 0-1), present only when diagnosis is Tuberculosis.")
    heatmap_base64: Optional[str] = Field(None, description="Base64-encoded PNG Grad-CAM heatmap with bounding box overlay, present only when diagnosis is Tuberculosis.")
    clinical_note: Optional[str] = Field(None, description="Clinical note, present only when diagnosis is Tuberculosis.")