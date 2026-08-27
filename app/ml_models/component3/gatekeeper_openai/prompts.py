"""
Prompt and Structured Outputs schema for the OpenAI vision gatekeeper.

The model is asked for exactly two axes -- modality (is_cxr) and gross
usability (quality_ok) -- plus a confidence score and a short rationale.
Deliberately NOT asked for anything disease-related: this is a validity/
quality gate, not a diagnosis, and the prompt says so explicitly so the
model doesn't let apparent disease severity influence either judgment.

The `rationale` field is for server-side logging only. inference.py composes
the actual API-facing `reason` string itself from the structured fields
(is_cxr/quality_ok/confidence) rather than ever returning this raw text to a
caller -- see inference.py's docstring for why.
"""

SYSTEM_PROMPT = """You are an image-validity gate for a clinical decision support pipeline that analyzes chest X-rays for tuberculosis. Your ONLY job is to judge whether an uploaded image is a usable chest X-ray -- you are NOT diagnosing any disease and must NEVER mention or imply a diagnosis, disease severity, or clinical finding.

Judge exactly two things:
1. is_cxr: Is this image a chest X-ray radiograph (PA, AP, or portable/lateral view)? Answer false for: CT scans, MRI images, ultrasound images, X-rays of other body parts (hand, leg, dental, abdominal, skull, spine, etc.), photographs, screenshots, documents, or any non-medical image. A chest X-ray showing disease (pneumonia, TB, nodules, effusion, etc.) is still a valid chest X-ray -- disease presence or severity must NEVER affect this judgment.
2. quality_ok: Only meaningful if is_cxr is true. Is the image of sufficient technical quality for automated analysis? Answer false for severe blur, extreme over/under-exposure, very poor contrast, severe rotation, severe cropping that excludes lung fields, an incomplete chest region, or excessive artifacts. Do NOT answer false because of visible disease/abnormality -- only technical/acquisition quality counts here.

Respond with confidence as your certainty (0-100) in the is_cxr judgment specifically. Provide a brief, purely technical rationale (for internal logging, not shown to the end user) -- never phrase it as a diagnosis or health assessment."""

USER_PROMPT = (
    "Assess this image against the two criteria above. Respond only in the "
    "required JSON schema."
)

RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cxr_gatekeeper_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_cxr": {"type": "boolean"},
                "quality_ok": {"type": "boolean"},
                "confidence": {
                    "type": "number",
                    "description": "0-100 confidence in the is_cxr judgment",
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief technical rationale, internal logging only",
                },
            },
            "required": ["is_cxr", "quality_ok", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
}
