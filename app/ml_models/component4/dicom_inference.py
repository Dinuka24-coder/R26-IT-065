import numpy as np

from app.ml_models.component4.dicom_model import get_dicom_model
from app.ml_models.component4.inference import preprocess

# Reuses inference.py's preprocess() UNCHANGED - it already does exactly
# the "resize 224x224 (INTER_NEAREST) -> normalize /255" steps that are
# the final stage of the DICOM preprocessing pipeline too. The
# DICOM-specific steps (HU conversion, windowing, grayscale, 3-channel
# replication) already happen upstream in
# app/ml_models/component4/dicom/renderer.py before these bytes ever
# reach this function - by the time image_bytes gets here, it's a PNG
# indistinguishable in format from a PNG/JPG upload. This is why no
# second preprocessing implementation was created: the shared final
# step already existed and is reused as-is.

# Confirmed against the actual trained model file (verified: input
# shape (None,224,224,3), output shape (None,4), texture_conv_4 layer
# present) and against dicom_preprocessing.py's CLASS_NAMES from the
# training script. The .keras file itself carries no label metadata -
# this order is only correct if it matches what the training script
# used, which was independently re-confirmed against training code
# before this integration.
DICOM_CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "small.cell.carcinoma",
    "squamous.cell.carcinoma",
]


def predict_dicom(image_bytes: bytes) -> dict:
    model = get_dicom_model()

    img = preprocess(image_bytes)

    prediction = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = DICOM_CLASS_NAMES[predicted_index]

    confidence = float(np.max(prediction))

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "raw_scores": {
            DICOM_CLASS_NAMES[i]: round(float(score), 4)
            for i, score in enumerate(prediction)
        }
    }