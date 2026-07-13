def classify_urgency(confidence: float, prediction: str) -> str:

    if prediction == "normal":
        return "Low"

    if confidence >= 85:
        return "High"

    elif confidence >= 65:
        return "Moderate"

    else:
        return "Low"