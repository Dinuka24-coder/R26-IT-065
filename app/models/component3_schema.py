from typing import Optional
from pydantic import BaseModel, Field

class TBPredictionResponse(BaseModel):
    database_record_id: Optional[str] = Field(None, description="The MongoDB document ID.")
    patient_id: Optional[str] = Field(None, description="The associated patient ID.")
    filename: Optional[str] = Field(None, description="The original uploaded filename.")
    prediction: str = Field(..., description="The textual diagnosis label.")
    confidence: float = Field(..., description="Confidence percentage out of 100.")
    raw_score: float = Field(..., description="Raw sigmoid output from the model.")
    requires_clinical_review: bool = Field(..., description="Flag indicating if the AI is unsure.")
