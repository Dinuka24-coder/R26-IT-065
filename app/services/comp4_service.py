from datetime import datetime, timezone

from app.ml_models.component4.inference import predict
from app.ml_models.component4.dicom_inference import predict_dicom
from app.ml_models.component4.gradcam import generate_gradcam
from app.ml_models.component4.dicom_model import get_dicom_model
from app.ml_models.component4.lung_ct_validation import check_lung_ct_suitability
from app.ml_models.component4.mobilenet_ood import check_mobilenet_ood
from app.ml_models.component4.segmentation import run_segmentation
from app.ml_models.component4.dicom.series import get_series_store, group_by_acquisition
from app.ml_models.component4.dicom.windowing import resolve_window
from app.ml_models.component4.dicom.renderer import render_slice_to_png_bytes
from app.ml_models.component4.dicom.volume import build_volume, VolumeGeometryError
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


class MultipleAcquisitionsError(ValueError):
    """Raised when a volume is requested WITHOUT specifying which
    acquisition, but the series genuinely contains more than one
    acquisition candidate. Never silently resolved by picking one - the
    caller must specify acquisition_number explicitly, per the approved
    architecture's constraint against automatic selection.

    Carries the real list of available acquisitions (as already-computed
    dicts from get_series_acquisitions()) so the API layer can return a
    useful, actionable error rather than a bare message.
    """

    def __init__(self, series_id: str, acquisitions: list[dict]):
        self.series_id = series_id
        self.acquisitions = acquisitions
        super().__init__(
            f"Series \'{series_id}\' contains {len(acquisitions)} acquisitions - "
            f"an acquisition_number must be specified explicitly."
        )


async def run_prediction(patient_id: str, image_bytes: bytes) -> dict:
    """Existing PNG/JPG prediction path.

    Now runs TWO suitability gates in sequence before classification:
      1. check_lung_ct_suitability() - existing Level 2 pixel-statistics
         check (aspect ratio, intensity std, color saturation).
      2. check_mobilenet_ood() - NEW, feature-space suitability check
         (MobileNetV2 centroid distance). Added because Level 2 alone
         cannot distinguish a chest X-ray from a lung CT slice - both
         are grayscale, reasonably shaped, real medical images. Uses
         the already-calibrated, already-verified centroid/threshold
         unchanged - see mobilenet_ood.py.

    Either gate can reject; both map to the same LungCtSuitabilityError
    -> same 4xx "unsuitable for this component" response the API layer
    already handles. No new response structure, no new exception type.

    Everything after both gates - predict(), generate_gradcam(), the
    result shape, save_result() - is UNCHANGED.
    """
    # --- Stage: Lung CT suitability validation (Level 2) ---
    validation_result = check_lung_ct_suitability(image_bytes)
    if not validation_result.is_valid:
        print("❌ Component 4 image REJECTED — Level 2 CT suitability validation")
        print(f"   Reasons: {validation_result.reasons}")
        raise LungCtSuitabilityError(validation_result.reasons)
    print("✅ Component 4 Level 2 validation PASSED")

    # --- Stage: Feature-space suitability / OOD validation (MobileNetV2) ---
    ood_result = check_mobilenet_ood(image_bytes)
    if not ood_result.is_valid:
        print("❌ Component 4 image REJECTED — MobileNetV2 OOD validation")
        print(f"   Distance: {ood_result.distance:.4f}")
        print(f"   Threshold: {ood_result.threshold}")
        print(f"   Reasons: {ood_result.reasons}")
        raise LungCtSuitabilityError(ood_result.reasons)
    print("✅ Component 4 MobileNetV2 OOD validation PASSED")

    print("🔬 Component 4 image PASSED all suitability checks — running cancer classification")

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

async def get_dicom_volume(series_id: str, acquisition_number: str | None = None):
    """MODIFIED (backward compatible) - now accepts an optional
    acquisition_number.

    When acquisition_number is None (the existing calling convention,
    unchanged for every existing caller):
      - if the series resolves to exactly ONE acquisition group (the
        521 real single-acquisition series, AND the 42 real
        missing-AcquisitionNumber series - both cases behave identically
        here, since group_by_acquisition() already collapses both to one
        group) - that one group is used automatically. EXACT existing
        behavior, byte-for-byte, for every series that does not need
        acquisition selection at all.
      - if the series resolves to MULTIPLE acquisition groups (the 90
        real multi-acquisition series), raises MultipleAcquisitionsError
        rather than silently choosing one - never guesses, per the
        approved architecture.

    When acquisition_number is provided, selects that specific group by
    matching AcquisitionGroup.acquisition_number == acquisition_number
    (as a string) - raises ValueError if no such acquisition exists in
    this series.

    Reuses the exact same DicomSeriesNotFoundError already used
    elsewhere. VolumeGeometryError propagates unchanged from
    build_volume() - the router maps it to 422, exactly as before.

    build_volume() is called exactly once per invocation, on exactly
    ONE group's datasets - never merges groups, never called on more
    than the selected candidate acquisition.
    """
    store = get_series_store()
    series = store.get(series_id)
    if series is None:
        raise DicomSeriesNotFoundError(
            f"DICOM series '{series_id}' was not found or has expired. "
            f"Please re-upload."
        )

    groups = group_by_acquisition(series.datasets)

    if acquisition_number is None:
        if len(groups) == 1:
            target_group = groups[0]
        else:
            acquisitions = get_series_acquisitions(series_id)
            raise MultipleAcquisitionsError(series_id, acquisitions)
    else:
        matching = [
            g for g in groups if g.acquisition_number == str(acquisition_number)
        ]
        if not matching:
            raise ValueError(
                f"Acquisition '{acquisition_number}' not found in series "
                f"'{series_id}'."
            )
        target_group = matching[0]

    volume = build_volume(target_group.datasets)
    return volume

def get_series_acquisitions(series_id: str) -> list[dict]:
    """Returns each candidate acquisition for a series with its REAL
    build_volume() validity - never constructs/returns a full DicomVolume
    here, only reports accept/reject + why, so this stays cheap to call
    for listing purposes.

    Reports only FACTUAL information already present in the DICOM tags
    or computed by build_volume() - never assigns clinical meaning
    (no "contrast phase", no "diagnostic", no "best" acquisition).
    AcquisitionNumber=1 is reported as exactly that: the value "1",
    nothing more.

    build_volume() is called once per group - the REAL, unmodified
    function, exactly as it already validates a whole series today.
    """
    store = get_series_store()
    series = store.get(series_id)
    if series is None:
        raise DicomSeriesNotFoundError(
            f"DICOM series \'{series_id}\' was not found or has expired. "
            f"Please re-upload."
        )

    groups = group_by_acquisition(series.datasets)

    results = []
    for group in groups:
        acquisition_times = sorted(set(
            str(getattr(ds, "AcquisitionTime")) for ds in group.datasets
            if hasattr(ds, "AcquisitionTime")
        ))
        try:
            build_volume(group.datasets)
            valid = True
            rejection_reason = None
        except VolumeGeometryError as exc:
            valid = False
            rejection_reason = str(exc)

        results.append({
            "acquisition_number": group.acquisition_number,
            "slice_count": len(group.datasets),
            "acquisition_time": acquisition_times,
            "valid": valid,
            "rejection_reason": rejection_reason,
        })

    return results