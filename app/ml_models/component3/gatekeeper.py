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
        """Reproduces the exact preprocessing used to build the centroid."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image")
        img_resized = cv2.resize(img, self.CENTROID_SIZE)
        return img_resized.astype('float64') / 255.0

    def inspect_image(self, image_bytes):
        """Returns (is_valid_cxr: bool, message: str)."""
        try:
            feature = self._extract_feature(image_bytes)
        except ValueError as e:
            return False, str(e)

        distance = float(np.linalg.norm(feature - self.centroid))

        if distance <= self.threshold:
            return True, f"Valid chest X-ray (distance={distance:.4f}, threshold={self.threshold})"

        return False, (
            f"Rejected: image does not match expected chest X-ray structure "
            f"(distance={distance:.4f}, threshold={self.threshold})"
        )
