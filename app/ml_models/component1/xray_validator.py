import numpy as np
import cv2

# ── Reference X-ray feature profile ────────────────────────────
# These values represent typical chest X-ray characteristics
# Computed from the training dataset statistics
XRAY_REFERENCE = {
    "mean_intensity":     0.45,
    "std_intensity":      0.24,
    "grayscale_ratio":    0.98,
    "vertical_symmetry":  0.75,
    "edge_density":       0.12,
    "dark_corner_ratio":  0.65,
    "central_darkness":   0.55,   # chest: dark lungs → high value
    "vertical_gradient":  0.15,   # chest: brighter at bottom
    "bright_ratio":       0.05,   # chest: few very-bright pixels
}

# Distance threshold — tuned empirically
DISTANCE_THRESHOLD = 0.60

# Chest X-rays are roughly square (0.7 – 1.4)
MIN_ASPECT_RATIO = 0.70
MAX_ASPECT_RATIO = 1.45


def check_aspect_ratio(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0
    h, w = img.shape
    ratio = w / h
    return (MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO), round(ratio, 3)

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

    # ── Feature 7: Central lung darkness ───────────────────────
    # Chest X-rays have DARK lung fields in the upper-middle region.
    # Dental panos have BRIGHT teeth/bone there.
    upper_middle = img_norm[40:120, 60:164]
    central_darkness = float(1.0 - upper_middle.mean())

    # ── Feature 8: Vertical intensity gradient ─────────────────
    # Chest: dark lungs on top, brighter abdomen/diaphragm below.
    top_third = img_norm[:75, :].mean()
    bottom_third = img_norm[149:, :].mean()
    vertical_gradient = float(bottom_third - top_third)

    # ── Feature 9: Bright pixel ratio ──────────────────────────
    # Dental images have many very bright pixels (enamel, fillings).
    bright_ratio = float(np.sum(img_norm > 0.75) / img_norm.size)

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "grayscale_ratio": grayscale_ratio,
        "vertical_symmetry": symmetry,
        "edge_density": edge_density,
        "dark_corner_ratio": dark_corner_ratio,
        "central_darkness": central_darkness,  # NEW
        "vertical_gradient": vertical_gradient,  # NEW
        "bright_ratio": bright_ratio,  # NEW
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
    try:
        # ── Gate 1: aspect ratio ────────────────────────────
        ratio_ok, ratio = check_aspect_ratio(image_bytes)
        if not ratio_ok:
            print(f"❌ Rejected: aspect ratio {ratio} outside chest X-ray range")
            return {
                "is_xray": False,
                "reason": f"Image proportions ({ratio}) don't match a chest X-ray. "
                          f"This looks like a different type of scan.",
                "aspect_ratio": ratio,
            }

        # ── Gate 2: feature distance ────────────────────────
        features = extract_features(image_bytes)
        distance = compute_euclidean_distance(features)
        is_valid = distance <= DISTANCE_THRESHOLD

        print(f"🔍 distance={distance:.4f} ratio={ratio} → {'PASS' if is_valid else 'REJECT'}")

        return {
            "is_xray":      is_valid,
            "distance":     round(distance, 4),
            "threshold":    DISTANCE_THRESHOLD,
            "aspect_ratio": ratio,
            "confidence":   round(max(0, min(100, (1 - distance/DISTANCE_THRESHOLD) * 100)), 2),
            "features":     {k: round(v, 3) for k, v in features.items()},
        }

    except Exception as e:
        return {"is_xray": False, "error": str(e)}