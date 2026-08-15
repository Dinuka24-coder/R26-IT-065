from datetime import datetime, timezone

from app.ml_models.component4.inference import predict
from app.ml_models.component4.dicom_inference import predict_dicom
from app.ml_models.component4.gradcam import generate_gradcam
from app.ml_models.component4.dicom_model import get_dicom_model
from app.ml_models.component4.lung_ct_validation import check_lung_ct_suitability
from app.ml_models.component4.segmentation import run_segmentation
from app.ml_models.component4.dicom.series import get_series_store
from app.ml_models.component4.dicom.windowing import resolve_window
from app.ml_models.component4.dicom.renderer import render_slice_to_png_bytes
from app.repositories.result_repo import save_result


class LungCtSuitabilityError(ValueError):
    """Raised when an input fails the Level 2 lung CT suitability check.

    This means "input is unsuitable for this component" — it must never
    be interpreted or displayed as a clinical finding (e.g. "Normal").
    Callers (the API layer) should map this to a 4xx response with the
    validation's user_message, not a prediction result.
    """

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class DicomSeriesNotFoundError(ValueError):
    """Raised when a series_id is unknown or has expired from the cache."""


async def run_prediction(patient_id: str, image_bytes: bytes) -> dict:
    """Existing PNG/JPG prediction path.

    UNCHANGED except for one addition: the lung CT suitability gate
    (Level 2 validation) now runs before classification, per the
    requirement that PNG/JPG get the same gate as DICOM. Nothing else
    about this function's behavior has changed.
    """
    # --- Stage: Lung CT suitability validation (Level 2) ---
    validation_result = check_lung_ct_suitability(image_bytes)
    if not validation_result.is_valid:
        raise LungCtSuitabilityError(validation_result.reasons)

    # --- Stage: Cancer classification (unchanged) ---
    result = predict(image_bytes)

    # --- Stage: Grad-CAM explanation (unchanged) ---
    heatmap_url = generate_gradcam(image_bytes)

    final_result = {
        "patient_id": patient_id,
        "component": "CT-Based-Lung-cancer-Classification",
        "input_type": "PNG_JPG",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "heatmap_url": heatmap_url,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    saved_id = await save_result(
        "lung_cancer_results",
        final_result
    )

    final_result["result_id"] = str(saved_id)

    return final_result


async def run_dicom_prediction(
    patient_id: str,
    series_id: str,
    slice_index: int,
    window_center: float | None,
    window_width: float | None,
    preset: str | None,
) -> dict:
    """DICOM slice prediction path.

    DICOM series -> select one 2D slice -> HU conversion -> windowing
    -> render to 2D image -> [same stages as PNG path from here on]

    Uses the DEDICATED DICOM-trained model (predict_dicom(), separate
    weights file, separate class order - no "normal" class, has
    "small.cell.carcinoma") rather than the original PNG/JPG model.
    generate_gradcam() is the same function used by the PNG/JPG path,
    just called with the DICOM model passed in explicitly - see
    gradcam.py's backward-compatible model= parameter.
    """
    store = get_series_store()
    series = store.get(series_id)
    if series is None:
        raise DicomSeriesNotFoundError(
            f"DICOM series '{series_id}' was not found or has expired. "
            f"Please re-upload."
        )

    if not (0 <= slice_index < series.number_of_slices):
        raise ValueError(
            f"Invalid slice_index {slice_index}; series has "
            f"{series.number_of_slices} slices."
        )

    slice_dataset = series.datasets[slice_index]

    # --- Stage: HU conversion + windowing ---
    resolved_wc, resolved_ww, window_source = resolve_window(
        preset=preset,
        window_center=window_center,
        window_width=window_width,
        dataset=slice_dataset,
    )
    rendered_png_bytes = render_slice_to_png_bytes(
        slice_dataset, window_center=resolved_wc, window_width=resolved_ww
    )

    # --- Stage: Lung CT suitability validation (Level 2) ---
    # Same gate, same function, as the PNG/JPG path — operates on the
    # rendered 2D bytes, agnostic to DICOM origin.
    validation_result = check_lung_ct_suitability(rendered_png_bytes)
    if not validation_result.is_valid:
        raise LungCtSuitabilityError(validation_result.reasons)

    # --- Stage: Cancer classification (DICOM model - separate from PNG/JPG) ---
    result = predict_dicom(rendered_png_bytes)

    # --- Stage: Grad-CAM explanation (same function, DICOM model passed in) ---
    heatmap_url = generate_gradcam(rendered_png_bytes, model=get_dicom_model())

    # --- Stage: Segmentation (extension point - not yet trained) ---
    # Called with the SAME rendered_png_bytes as classification/Grad-CAM
    # above, so if/when a real model is wired in, it analyzes the exact
    # same slice - no separate fetch, no risk of a different slice.
    segmentation_result = run_segmentation(rendered_png_bytes)

    final_result = {
        "patient_id": patient_id,
        "component": "CT-Based-Lung-cancer-Classification",
        "input_type": "DICOM",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "heatmap_url": heatmap_url,
        "slice_index": slice_index,
        "total_slices": series.number_of_slices,
        "window_center": resolved_wc,
        "window_width": resolved_ww,
        "window_source": window_source,
        "segmentation": {
            "available": segmentation_result.available,
            "reason": segmentation_result.reason,
            "mask_url": segmentation_result.mask_url,
            "overlay_url": segmentation_result.overlay_url,
        },
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    saved_id = await save_result(
        "lung_cancer_results",
        final_result
    )

    final_result["result_id"] = str(saved_id)

    return final_result


async def run_dicom_volume_prediction(
    patient_id: str,
    series_id: str,
    preset: str | None,
    window_center: float | None,
    window_width: float | None,
) -> dict:
    """Volume-level aggregation over ALL slices in a DICOM series.

    Runs the DEDICATED DICOM-trained model (predict_dicom()) once per
    slice - updated from the original PNG/JPG model now that a real
    DICOM-trained model exists; using the PNG model here would have
    been a real mismatch (wrong class order, wrong training
    distribution) now that a dedicated one is available.
    """
    store = get_series_store()
    series = store.get(series_id)
    if series is None:
        raise DicomSeriesNotFoundError(
            f"DICOM series '{series_id}' was not found or has expired. "
            f"Please re-upload."
        )

    per_slice_results = []
    skipped_slices = []

    for slice_index, slice_dataset in enumerate(series.datasets):
        resolved_wc, resolved_ww, window_source = resolve_window(
            preset=preset,
            window_center=window_center,
            window_width=window_width,
            dataset=slice_dataset,
        )
        rendered_png_bytes = render_slice_to_png_bytes(
            slice_dataset, window_center=resolved_wc, window_width=resolved_ww
        )

        validation_result = check_lung_ct_suitability(rendered_png_bytes)
        if not validation_result.is_valid:
            skipped_slices.append({
                "slice_index": slice_index,
                "reasons": validation_result.reasons,
            })
            continue

        result = predict_dicom(rendered_png_bytes)
        per_slice_results.append({
            "slice_index": slice_index,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_scores": result["raw_scores"],
        })

    if not per_slice_results:
        return {
            "patient_id": patient_id,
            "component": "CT-Based-Lung-cancer-Classification",
            "input_type": "DICOM_VOLUME",
            "total_slices": series.number_of_slices,
            "analyzed_slices": 0,
            "skipped_slices": skipped_slices,
            "aggregation_method": None,
            "prediction": None,
            "confidence": None,
            "per_slice_results": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # Mean softmax aggregation across analyzed slices.
    class_names = list(per_slice_results[0]["raw_scores"].keys())
    mean_scores = {
        cls: sum(r["raw_scores"][cls] for r in per_slice_results) / len(per_slice_results)
        for cls in class_names
    }
    aggregated_prediction = max(mean_scores, key=mean_scores.get)
    aggregated_confidence = round(mean_scores[aggregated_prediction] * 100, 2)

    return {
        "patient_id": patient_id,
        "component": "CT-Based-Lung-cancer-Classification",
        "input_type": "DICOM_VOLUME",
        "total_slices": series.number_of_slices,
        "analyzed_slices": len(per_slice_results),
        "skipped_slices": skipped_slices,
        "aggregation_method": "mean_softmax",
        "prediction": aggregated_prediction,
        "confidence": aggregated_confidence,
        "mean_scores": {k: round(v, 4) for k, v in mean_scores.items()},
        "per_slice_results": per_slice_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }