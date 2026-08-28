from typing import List, Optional
from pydantic import BaseModel, Field


class TBPredictionResponse(BaseModel):
    database_record_id: Optional[str] = Field(None, description="The MongoDB document ID.")
    patient_id: Optional[str] = Field(None, description="The associated patient ID.")
    filename: Optional[str] = Field(None, description="The original uploaded filename.")
    status: str = Field(..., description="'success' if the scan was processed, 'rejected' if the gatekeeper turned the upload away (not a chest X-ray, or quality too low).")
    diagnosis: Optional[str] = Field(None, description="Healthy, Non-TB, or Tuberculosis. None when status is 'rejected'.")
    confidence_score: Optional[float] = Field(None, description="Confidence percentage out of 100. None when status is 'rejected'.")
    message: Optional[str] = Field(None, description="Human-readable status message.")
    bounding_box: Optional[List[float]] = Field(None, description="TB localization box [x_min, y_min, x_max, y_max] (normalized 0-1), present only when diagnosis is Tuberculosis.")
    heatmap_base64: Optional[str] = Field(None, description="Base64-encoded PNG Grad-CAM heatmap with bounding box overlay, present only when diagnosis is Tuberculosis.")
    clinical_note: Optional[str] = Field(None, description="Clinical note, present only when diagnosis is Tuberculosis.")
    is_cxr: Optional[bool] = Field(None, description="Whether the gatekeeper judged the upload to be a chest X-ray. Independent of diagnosis.")
    cxr_confidence: Optional[float] = Field(None, description="Gatekeeper confidence (0-100) that the upload is a chest X-ray.")
    quality_score: Optional[float] = Field(None, description="Gatekeeper image-usability score (0-100), None when is_cxr is False (undefined for non-CXR content).")
    gatekeeper_backend: Optional[str] = Field(None, description="Which gatekeeper stage(s) produced the decision: 'heuristic', 'heuristic+openai'/'openai', 'heuristic+cnn_fallback'/'cnn_fallback' (used when OpenAI is unconfigured or its call failed), or 'heuristic_only_cnn_unavailable'.")


class GatekeeperResponse(BaseModel):
    """Standalone gatekeeper-only response. Never implies a TB/clinical
    diagnosis -- it only answers whether an upload is a valid, usable chest
    X-ray suitable for downstream automated analysis."""
    filename: Optional[str] = Field(None, description="The original uploaded filename.")
    accepted: bool = Field(..., description="True if the image is a valid, usable chest X-ray.")
    is_cxr: bool = Field(..., description="Whether the upload was judged to be a chest X-ray at all.")
    cxr_confidence: float = Field(..., description="Confidence (0-100) that the upload is a chest X-ray.")
    quality_score: Optional[float] = Field(None, description="Image-usability score (0-100); None when is_cxr is False.")
    reason: str = Field(..., description="Human-readable explanation. Never implies a TB/clinical diagnosis.")
    gatekeeper_backend: str = Field(..., description="Which gatekeeper stage(s) produced the decision: 'heuristic', 'heuristic+openai'/'openai', 'heuristic+cnn_fallback'/'cnn_fallback', or 'heuristic_only_cnn_unavailable'.")
    gatekeeper_heatmap_base64: Optional[str] = Field(None, description="Base64-encoded PNG Grad-CAM attention map for the gatekeeper's CXR decision. NOT TB evidence. Only ever populated when the local CNN actually ran (gatekeeper_backend contains 'cnn_fallback') -- OpenAI's hosted model exposes no gradients, so no heatmap is possible on that path.")
