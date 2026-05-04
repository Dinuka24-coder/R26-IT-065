def calculate_pneumonia_severity(confidence_score):
    """Calculates a qualitative severity level based on the AI confidence score."""
    if confidence_score < 50:
        return "N/A (Normal)"
    elif 50 <= confidence_score < 70:
        return "Mild / Early Stage"
    elif 70 <= confidence_score < 90:
        return "Moderate"
    else:
        return "Severe"

