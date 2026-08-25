import numpy as np


def calculate_pneumonia_severity(confidence_score):
    """Calculates a qualitative severity level based on the AI confidence score."""
    if confidence_score < 47:
        return "N/A (Normal)"
    elif 47 <= confidence_score < 70:
        return "Mild / Early Stage"
    elif 70 <= confidence_score < 90:
        return "Moderate"
    else:
        return "Severe"


def calculate_heatmap_severity(heatmap):
    """Computes spatial severity metrics from a Grad-CAM heatmap.
    
    Args:
        heatmap: 2D numpy array with values in [0, 1] from Grad-CAM.
        
    Returns:
        dict with 'affected_area_percent' and 'mean_intensity' keys.
    """
    # Regions where heatmap activation exceeds 0.5
    activated_mask = heatmap > 0.5
    total_pixels = heatmap.size

    affected_area_percent = float(np.sum(activated_mask) / total_pixels * 100)
    
    if np.any(activated_mask):
        mean_intensity = float(np.mean(heatmap[activated_mask]))
    else:
        mean_intensity = 0.0
    
    return {
        "affected_area_percent": round(affected_area_percent, 2),
        "mean_intensity": round(mean_intensity, 4)
    }
