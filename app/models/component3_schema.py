from pydantic import BaseModel, Field

class TBPredictionResponse(BaseModel):
    prediction: str = Field(..., description="The textual diagnosis label.")
    confidence: float = Field(..., description="Confidence percentage out of 100.")
    raw_score: float = Field(..., description="Raw sigmoid output from the model.")
    requires_clinical_review: bool = Field(..., description="Flag indicating if the AI is unsure.")
