from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from app.services.comp4_service import (
    run_prediction,
    run_dicom_prediction,
    LungCtSuitabilityError,
    DicomSeriesNotFoundError,
)
from app.ml_models.component4.dicom.reader import read_dicom, DicomStructureError
from app.ml_models.component4.dicom.validation import validate_ct_modality, safe_public_metadata
from app.ml_models.component4.dicom.series import get_series_store
from app.ml_models.component4.dicom.windowing import resolve_window
from app.ml_models.component4.dicom.renderer import render_slice_preview


router = APIRouter()


@router.post("/predict")
async def predict_lung_cancer_sub_type(
    patient_id: str = Form(...),
    file: UploadFile = File(...)
):
    """UNCHANGED endpoint contract. Internally now runs the lung CT
    suitability gate before classification (see comp4_service.py);
    behavior for valid lung CT input is identical to before.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    image_bytes = await file.read()

    try:
        result = await run_prediction(
            patient_id,
            image_bytes
        )
    except LungCtSuitabilityError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported image. This component accepts lung CT images "
                "only. Input is unsuitable for this component."
            ),
        ) from exc

    return result


@router.post("/dicom/inspect")
async def inspect_dicom(
    files: list[UploadFile] = File(...),
):
    """Accepts one or more DICOM files (a single slice, or a full
    series). Validates structure and CT modality (Level 1 only —
    lung suitability is checked later, at analyze time, on the
    doctor-selected/windowed slice). Returns non-PHI metadata and a
    server-generated series_id for subsequent /dicom/slice and
    /dicom/analyze calls.
    """
    parsed_files = []
    for f in files:
        raw = await f.read()
        try:
            parsed = read_dicom(raw)
        except DicomStructureError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        parsed_files.append(parsed)

    for pf in parsed_files:
        ct_check = validate_ct_modality(pf.dataset)
        if not ct_check.is_valid:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported medical image. This component accepts CT "
                    "DICOM images only. " + " ".join(ct_check.reasons)
                ),
            )

    store = get_series_store()
    series = store.create(parsed_files)

    metadata = safe_public_metadata(series.datasets[0])

    return {
        "series_id": series.series_id,
        "number_of_slices": series.number_of_slices,
        **metadata,
    }


@router.get("/dicom/{series_id}/slice/{slice_index}")
async def get_dicom_slice(
    series_id: str,
    slice_index: int,
    preset: Optional[str] = None,
    wc: Optional[float] = None,
    ww: Optional[float] = None,
):
    """Returns a rendered PNG of the requested slice for the viewer to
    display, with the requested window/level applied. This is the
    viewer PREVIEW image — the same rendering logic is used again at
    analyze time on whatever slice/window the doctor has settled on
    when they click "Analyze Current Slice".
    """
    store = get_series_store()
    series = store.get(series_id)
    if series is None:
        raise HTTPException(
            status_code=404,
            detail="DICOM series not found or has expired. Please re-upload.",
        )

    if not (0 <= slice_index < series.number_of_slices):
        raise HTTPException(status_code=400, detail="Invalid slice_index.")

    dataset = series.datasets[slice_index]

    try:
        resolved_wc, resolved_ww, _source = resolve_window(
            preset=preset, window_center=wc, window_width=ww, dataset=dataset
        )
        png_bytes = render_slice_preview(
            dataset, window_center=resolved_wc, window_width=resolved_ww
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=png_bytes, media_type="image/png")


@router.post("/dicom/analyze")
async def analyze_dicom_slice(
    patient_id: str = Form(...),
    series_id: str = Form(...),
    slice_index: int = Form(...),
    window_center: Optional[float] = Form(None),
    window_width: Optional[float] = Form(None),
    preset: Optional[str] = Form(None),
):
    """Doctor-triggered analysis of the currently selected/windowed
    slice. Explicit action only — never auto-triggered on upload or
    on slice navigation.
    """
    try:
        result = await run_dicom_prediction(
            patient_id=patient_id,
            series_id=series_id,
            slice_index=slice_index,
            window_center=window_center,
            window_width=window_width,
            preset=preset,
        )
    except DicomSeriesNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LungCtSuitabilityError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported image. This component accepts lung CT images "
                "only. Input is unsuitable for this component."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result