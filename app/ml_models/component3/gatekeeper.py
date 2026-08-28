import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - GATEKEEPER - %(message)s')


class EuclideanCXRGatekeeper:
    """
    Rejects uploads that are not structurally similar to a chest X-ray.

    Matches the centroid-generation approach used to build master_cxr_centroid.npy:
    each image is decoded as grayscale, resized to 64x64, and normalized to [0, 1].
    The gatekeeper measures the Euclidean distance between that image's pixel
    array and the pre-computed centroid (the pixel-wise mean of the training
    chest X-ray set). A large distance means the image's overall structure/
    exposure doesn't resemble a chest X-ray.
    """

    CENTROID_SIZE = (64, 64)

    # Calibrated against all 12,278 real chest X-rays bundled in
    # training/TBX11K/imgs (same preprocessing as _extract_feature):
    # distance distribution had mean=11.83, std=2.31, p99=18.37, p99.5=21.54,
    # and a Tukey extreme-outlier fence (Q3 + 3*IQR) of 22.16. The old
    # placeholder of 14.0 sat inside the normal range and rejected ~15.6% of
    # genuine CXRs (e.g. wide-frame/portable views with more shoulder, arm,
    # and abdomen in shot than a tightly-cropped PA film).
    #
    # NOTE: this is a single global centroid over raw pixel intensity, so it
    # mainly captures overall exposure/framing, not anatomy -- it cannot
    # reliably tell a chest film apart from another similarly-exposed body
    # region (e.g. an abdominal/KUB X-ray), since both are grayscale
    # radiographs with a comparable brightness distribution. 18.0 (~p99)
    # trades a slightly higher false-rejection rate on real CXRs (~1.2%,
    # vs ~0.4% at 22.0) for tighter rejection of those non-chest films; it
    # cannot eliminate that failure mode by itself. A durable fix needs a
    # feature that's actually chest-specific (e.g. lung-field symmetry) or
    # a small classifier, rather than pure distance-to-centroid.
    DEFAULT_THRESHOLD = 18.0

    # Real CXRs (all 12,278 TBX11K training images, same preprocessing) never
    # go below std=0.156 at 64x64 -- there's always rib/mediastinum/soft-tissue
    # structure. Distance-to-centroid alone can't catch a flat/blank/solid-color
    # image because such an image can still land near the centroid's mean
    # brightness (~0.5) and pass on exposure alone despite having zero internal
    # structure. Floor set well below the observed real-CXR minimum (0.156) so
    # this only catches genuinely degenerate uploads, never a real film.
    MIN_STD = 0.08

    # Pure per-pixel noise passes both the distance and std checks above --
    # its brightness average and variance both happen to fall inside the
    # real-CXR range, but it has no coherent anatomy. Edge density (Canny,
    # same 64x64 frame) is the discriminator that actually catches it: real
    # CXRs (800-image sample from TBX11K, same preprocessing) top out at
    # 0.299 (p99.5=0.282), because rib/soft-tissue edges are locally
    # correlated, whereas uncorrelated pixel noise produces edges nearly
    # everywhere (observed 0.36-0.38 across five random-noise samples).
    # Threshold set at the midpoint of that gap.
    MAX_EDGE_DENSITY = 0.32

    # Chest X-rays are roughly left-right mirror symmetric (same 800-image
    # TBX11K sample, same 64x64 frame: observed min=0.652, p1=0.739). This
    # doesn't catch uniform noise (its symmetry score lands in-range by
    # chance, ~0.78-0.79) but it's a cheap second net against otherwise
    # structured but clearly non-anatomical content that slips past the
    # edge-density and distance checks. Floor set below the observed
    # real-CXR minimum for safety margin.
    MIN_SYMMETRY = 0.55

    def __init__(self, centroid_path=None, threshold=None):
        if centroid_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            centroid_path = os.path.join(current_dir, 'weights', 'master_cxr_centroid.npy')
        self.centroid_path = centroid_path
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.centroid = self._load_centroid()

    def _load_centroid(self):
        if not os.path.exists(self.centroid_path):
            logging.error(f"FATAL: Centroid file not found at {self.centroid_path}")
            raise FileNotFoundError(f"Master CXR centroid not found: {self.centroid_path}")
        centroid = np.load(self.centroid_path)
        logging.info(f"Loaded master CXR centroid {centroid.shape} from {self.centroid_path}")
        return centroid

    def _extract_feature(self, image_bytes):
        """Reproduces the exact preprocessing used to build the centroid.
        Returns (normalized_float64_feature, resized_uint8_image) -- the
        latter is reused for the structural (edge/symmetry) checks so the
        image is only decoded and resized once."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image")
        img_resized = cv2.resize(img, self.CENTROID_SIZE)
        return img_resized.astype('float64') / 255.0, img_resized

    @staticmethod
    def _edge_density(img_resized_uint8):
        edges = cv2.Canny(img_resized_uint8, 50, 150)
        return float(np.sum(edges > 0) / edges.size)

    @staticmethod
    def _vertical_symmetry(img_resized_uint8):
        norm = img_resized_uint8.astype('float64') / 255.0
        half = norm.shape[1] // 2
        left = norm[:, :half]
        right = np.fliplr(norm[:, half:])
        return float(1.0 - np.mean(np.abs(left - right)))

    def inspect_image(self, image_bytes):
        """Returns (is_valid_cxr: bool, message: str)."""
        try:
            feature, img_resized = self._extract_feature(image_bytes)
        except ValueError as e:
            return False, str(e)

        if float(feature.std()) < self.MIN_STD:
            return False, (
                f"Rejected: image has no internal structure "
                f"(std={float(feature.std()):.4f} < {self.MIN_STD}), not a chest X-ray"
            )

        edge_density = self._edge_density(img_resized)
        if edge_density > self.MAX_EDGE_DENSITY:
            return False, (
                f"Rejected: image has no coherent anatomical structure "
                f"(edge_density={edge_density:.4f} > {self.MAX_EDGE_DENSITY}), not a chest X-ray"
            )

        symmetry = self._vertical_symmetry(img_resized)
        if symmetry < self.MIN_SYMMETRY:
            return False, (
                f"Rejected: image lacks the left-right symmetry of a chest X-ray "
                f"(symmetry={symmetry:.4f} < {self.MIN_SYMMETRY})"
            )

        distance = float(np.linalg.norm(feature - self.centroid))

        if distance <= self.threshold:
            return True, f"Valid chest X-ray (distance={distance:.4f}, threshold={self.threshold})"

        return False, (
            f"Rejected: image does not match expected chest X-ray structure "
            f"(distance={distance:.4f}, threshold={self.threshold})"
        )

    def inspect_image_detailed(self, image_bytes):
        """Same decision as inspect_image(), reshaped into the shared gatekeeper
        contract {is_cxr, cxr_confidence, quality_score, accepted, reason} so
        callers (controller.py's cascade) can treat this and the CNN gatekeeper
        polymorphically. This heuristic has no notion of image quality
        independent of CXR-likeness, so quality_score is always None."""
        accepted, reason = self.inspect_image(image_bytes)

        # Reverse-engineer a 0-100 confidence from the same distance figure
        # inspect_image() already computed, using the same "closer to threshold
        # is stronger evidence" formula xray_validator.py (component1) already
        # establishes as this codebase's confidence-scoring precedent. The
        # std/edge/symmetry guard rejections don't carry a distance value, so
        # they're reported at the extremes (0 confidence) instead.
        if "distance=" in reason:
            try:
                distance = float(reason.split("distance=")[1].split(",")[0])
                cxr_confidence = max(0.0, min(100.0, (1.0 - distance / self.threshold) * 100.0))
            except (IndexError, ValueError):
                cxr_confidence = 100.0 if accepted else 0.0
        else:
            cxr_confidence = 100.0 if accepted else 0.0

        return {
            "is_cxr": accepted,
            "cxr_confidence": round(cxr_confidence, 2),
            "quality_score": None,
            "accepted": accepted,
            "reason": reason,
        }
