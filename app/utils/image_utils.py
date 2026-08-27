import cv2
import numpy as np

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # generous headroom over typical CXR PNG/JPEG sizes

# Magic-byte prefixes for the image formats routes actually accept. Client-supplied
# Content-Type is trivially spoofable, so this is checked against the real bytes.
_MAGIC_PREFIXES = (
    b"\xff\xd8\xff",              # JPEG
    b"\x89PNG\r\n\x1a\n",         # PNG
    b"BM",                        # BMP
    b"II*\x00", b"MM\x00*",       # TIFF (little/big endian)
)


def validate_upload_bytes(image_bytes: bytes) -> None:
    """Raises ValueError if the bytes are empty, oversized, not a recognized image
    format by magic bytes, or fail to decode as an actual image."""
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"Uploaded file is too large ({len(image_bytes)} bytes); "
            f"max allowed is {MAX_UPLOAD_SIZE_BYTES} bytes."
        )
    if not any(image_bytes.startswith(sig) for sig in _MAGIC_PREFIXES):
        raise ValueError("Uploaded file is not a recognized image format (JPEG/PNG/BMP/TIFF).")

    nparr = np.frombuffer(image_bytes, np.uint8)
    if cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) is None:
        raise ValueError("Uploaded file could not be decoded as an image (corrupted or truncated).")


def safe_decode_image(image_bytes: bytes, mode=cv2.IMREAD_COLOR) -> np.ndarray:
    """validate_upload_bytes() + cv2.imdecode(). Use this instead of a raw
    cv2.imdecode call anywhere an upload's bytes are being decoded."""
    validate_upload_bytes(image_bytes)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, mode)
    if img is None:
        raise ValueError("Uploaded file could not be decoded as an image (corrupted or truncated).")
    return img
