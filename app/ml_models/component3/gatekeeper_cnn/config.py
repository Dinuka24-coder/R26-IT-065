IMG_SIZE = (224, 224)

CLASS_LABELS = ("valid_cxr", "non_cxr", "poor_quality_cxr")

# Operating-point thresholds for the two sigmoid heads. Placeholders until
# training/gatekeeper_cnn/evaluate_gatekeeper.py runs its threshold-selection
# sweep against the validation split (maximize is_cxr recall subject to
# precision >= 0.98) and overwrites these with the calibrated values -- see
# that script's docstring for the exact procedure. Do not hand-tune these
# without re-running the sweep; they must stay traceable to validation data.
CXR_THRESHOLD = 0.0296
QUALITY_THRESHOLD = 0.0003

WEIGHTS_FILENAME = "gatekeeper_cnn.keras"
