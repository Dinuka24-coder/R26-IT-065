def classify_urgency(confidence: float, prediction: str) -> str:
    # If NOT pneumothorax → always Low urgency
    if prediction != "Pneumothorax Detected":
        return "Low"

    # Only pneumothorax cases get urgency levels
    if confidence >= 85:
        return "High"
    elif confidence >= 65:
        return "Moderate"
    else:
        return "Low"