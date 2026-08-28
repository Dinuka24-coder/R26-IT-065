import base64
import json
import logging

import cv2
from openai import AsyncOpenAI

from app.ml_models.component3.gatekeeper_openai.config import (
    MODEL_NAME, MAX_TOKENS, TIMEOUT_SECONDS, SEND_IMAGE_MAX_SIDE, JPEG_QUALITY,
)
from app.ml_models.component3.gatekeeper_openai.prompts import (
    SYSTEM_PROMPT, USER_PROMPT, RESPONSE_JSON_SCHEMA,
)
from app.utils.image_utils import safe_decode_image


def _downscale_to_jpeg_data_url(image_bytes: bytes) -> str:
    """Downscales to SEND_IMAGE_MAX_SIDE on the long edge and re-encodes as
    JPEG -- a direct cost control (vision-API pricing scales with image
    size/tiles) that doesn't affect what this stage judges (modality/gross
    quality, not fine diagnostic detail)."""
    img = safe_decode_image(image_bytes, mode=cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = SEND_IMAGE_MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Failed to encode image for the OpenAI gatekeeper call.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class OpenAICXRGatekeeper:
    """Primary CXR/quality gate using an OpenAI vision model. Same shared
    contract as gatekeeper.py/gatekeeper_cnn -- {is_cxr, cxr_confidence,
    quality_score, accepted, reason} -- but async, since this is a real
    network call. controller.py catches any exception from
    inspect_image_detailed() and falls back to the local CNN+heuristic; this
    class itself does not retry or swallow errors (the SDK's own
    max_retries handles transient failures)."""

    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.client = AsyncOpenAI(api_key=api_key, timeout=TIMEOUT_SECONDS)
        self.model = model

    async def inspect_image_detailed(self, image_bytes: bytes) -> dict:
        data_url = _downscale_to_jpeg_data_url(image_bytes)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            response_format=RESPONSE_JSON_SCHEMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        verdict = json.loads(response.choices[0].message.content)

        is_cxr = bool(verdict["is_cxr"])
        quality_ok = bool(verdict["quality_ok"])
        confidence = max(0.0, min(100.0, float(verdict["confidence"])))
        # Logged server-side only -- never returned to a caller. See module
        # docstring / prompts.py: the API-facing `reason` below is always
        # composed by us from the structured fields, not echoed from the LLM,
        # so no unpredictable model phrasing (including accidental clinical
        # language) can reach a client.
        logging.info(
            "OpenAI gatekeeper verdict: is_cxr=%s quality_ok=%s confidence=%.2f rationale=%r",
            is_cxr, quality_ok, confidence, verdict.get("rationale", ""),
        )

        if not is_cxr:
            return {
                "is_cxr": False,
                "cxr_confidence": round(confidence, 2),
                "quality_score": None,
                "accepted": False,
                "reason": (
                    f"Rejected: image does not appear to be a chest X-ray "
                    f"(cxr_confidence={confidence:.2f}%), unsuitable for automated analysis."
                ),
            }

        if not quality_ok:
            return {
                "is_cxr": True,
                "cxr_confidence": round(confidence, 2),
                "quality_score": 0.0,
                "accepted": False,
                "reason": (
                    "Rejected: image quality is too low for reliable automated analysis "
                    "(blur, poor exposure, rotation, or an incomplete chest region). "
                    "This does not reflect on any diagnosis."
                ),
            }

        return {
            "is_cxr": True,
            "cxr_confidence": round(confidence, 2),
            "quality_score": 100.0,
            "accepted": True,
            "reason": (
                f"Accepted: valid chest X-ray suitable for downstream analysis "
                f"(cxr_confidence={confidence:.2f}%)."
            ),
        }
