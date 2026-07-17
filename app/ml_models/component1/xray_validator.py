import numpy as np
import cv2

# ── Reference X-ray feature profile ────────────────────────────
# These values represent typical chest X-ray characteristics
# Computed from the training dataset statistics
XRAY_REFERENCE = {
    "mean_intensity":     0.45,   # X-rays have medium-dark average
    "std_intensity":      0.24,   # good spread of light/dark
    "grayscale_ratio":    0.98,   # X-rays are nearly pure grayscale
    "vertical_symmetry":  0.75,   # chest is roughly symmetric L-R
    "edge_density":       0.12,   # moderate edges (ribs, organs)
    "dark_corner_ratio":  0.65,   # corners usually dark
}

# Distance threshold — tuned empirically
DISTANCE_THRESHOLD = 0.35


def extract_features(image_bytes: bytes) -> dict:
    """Extract statistical features from an uploaded image"""

    nparr = np.frombuffer(image_bytes, np.uint8)

    # Load color version to check if grayscale
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray  = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise ValueError("Could not decode image")

    img_resized = cv2.resize(img_gray, (224, 224))
    img_norm    = img_resized.astype(np.float32) / 255.0

    # ── Feature 1: Mean intensity ──────────────────────────────
    mean_intensity = float(np.mean(img_norm))

    # ── Feature 2: Std intensity ───────────────────────────────
    std_intensity = float(np.std(img_norm))

    # ── Feature 3: Grayscale ratio ─────────────────────────────
    # X-rays are grayscale — R, G, B channels nearly identical
    img_color_resized = cv2.resize(img_color, (224, 224))
    b, g, r = cv2.split(img_color_resized.astype(np.float32))
    channel_diff  = (np.abs(r - g).mean() + np.abs(g - b).mean()) / 2
    grayscale_ratio = float(1.0 - (channel_diff / 255.0))

    # ── Feature 4: Vertical symmetry ───────────────────────────
    # Chest X-rays are roughly left-right symmetric
    left_half   = img_norm[:, :112]
    right_half  = np.fliplr(img_norm[:, 112:])
    symmetry    = float(1.0 - np.mean(np.abs(left_half - right_half)))

    # ── Feature 5: Edge density ────────────────────────────────
    edges        = cv2.Canny(img_resized, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    # ── Feature 6: Dark corner ratio ───────────────────────────
    # X-ray corners are usually dark (outside body area)
    corners = [
        img_norm[:30, :30],      # top-left
        img_norm[:30, -30:],     # top-right
        img_norm[-30:, :30],     # bottom-left
        img_norm[-30:, -30:],    # bottom-right
    ]
    corner_mean      = np.mean([c.mean() for c in corners])
    dark_corner_ratio = float(1.0 - corner_mean)

    return {
        "mean_intensity":    mean_intensity,
        "std_intensity":     std_intensity,
        "grayscale_ratio":   grayscale_ratio,
        "vertical_symmetry": symmetry,
        "edge_density":      edge_density,
        "dark_corner_ratio": dark_corner_ratio,
    }


def compute_euclidean_distance(features: dict) -> float:
    """Compute Euclidean distance from reference X-ray profile"""

    # Build feature vectors in same order
    keys = list(XRAY_REFERENCE.keys())

    reference_vector = np.array([XRAY_REFERENCE[k] for k in keys])
    image_vector     = np.array([features[k]       for k in keys])

    # Euclidean distance
    distance = float(np.sqrt(np.sum((reference_vector - image_vector) ** 2)))

    return distance


def is_xray(image_bytes: bytes) -> dict:
    """
    Validate if uploaded image is a chest X-ray
    using Euclidean distance from reference profile
    """
    try:
        features = extract_features(image_bytes)
        distance = compute_euclidean_distance(features)

        is_valid = distance <= DISTANCE_THRESHOLD

        # Confidence score (inverse of distance, normalized)
        confidence = max(0, min(100, (1 - distance / DISTANCE_THRESHOLD) * 100))

        return {
            "is_xray":         is_valid,
            "distance":        round(distance, 4),
            "threshold":       DISTANCE_THRESHOLD,
            "confidence":      round(confidence, 2),
            "features":        {k: round(v, 3) for k, v in features.items()},
        }

    except Exception as e:
        return {
            "is_xray":  False,
            "distance": None,
            "error":    str(e),
        }