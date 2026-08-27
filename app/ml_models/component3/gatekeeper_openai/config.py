from app.config import settings

MODEL_NAME = settings.OPENAI_MODEL

# Small -- the response is a short structured-JSON verdict, not prose.
MAX_TOKENS = 200

# Applies to the API call itself, not the whole request lifecycle. Kept short
# because this sits in a live request path with a local fallback behind it --
# a hung call should fail fast into the CNN+heuristic fallback rather than
# leaving an upload hanging.
TIMEOUT_SECONDS = 15.0

# Vision-API pricing scales with image size/tiles, and this task only needs
# enough resolution to judge modality and gross quality issues (blur,
# exposure, rotation, missing anatomy) -- not fine diagnostic detail. Sending
# a downscaled JPEG instead of the original is a direct, deliberate cost
# control that doesn't affect what this stage is actually judging.
SEND_IMAGE_MAX_SIDE = 512
JPEG_QUALITY = 85
