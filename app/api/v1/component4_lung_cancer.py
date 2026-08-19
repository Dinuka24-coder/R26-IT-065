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
from app.ml_models.component4.dicom.volume import VolumeGeometryError
from app.services.comp4_service import (
    get_dicom_volume,
    get_series_acquisitions,
    MultipleAcquisitionsError,
)
import numpy as np


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

@router.get("/dicom/{series_id}/acquisitions")
async def list_dicom_acquisitions(series_id: str):
    """Lists every candidate acquisition within a series with its REAL
    build_volume() validity - never constructs/returns voxel data here.
    Calls the Stage 1 service function directly - no grouping logic
    duplicated in the router.

    Reports only factual, tag-derived information. Never assigns
    clinical meaning (no "contrast phase", no "best acquisition") -
    acquisition_number is reported exactly as the DICOM tag value,
    nothing more.
    """
    try:
        acquisitions = get_series_acquisitions(series_id)
    except DicomSeriesNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if len(acquisitions) == 1 and acquisitions[0]["acquisition_number"] is None:
        classification = "missing_acquisition_number"
    elif len(acquisitions) == 1:
        classification = "single_acquisition"
    else:
        classification = "multiple_acquisitions"

    return {
        "series_id": series_id,
        "classification": classification,
        "acquisitions": acquisitions,
    }


@router.get("/dicom/{series_id}/volume/metadata")
async def get_dicom_volume_metadata(series_id: str, acquisition_number: Optional[str] = None):
    """Small JSON metadata for the volume - NEVER voxel data. Uses the
    same series_id identity as the existing 2D endpoints. Deliberately
    excludes patient_id - only geometry needed by the frontend.

    acquisition_number is OPTIONAL, defaulting to None - existing
    callers (single-acquisition and missing-AcquisitionNumber series,
    the vast majority) see byte-identical behavior. For a genuinely
    multi-acquisition series, omitting it raises 400 rather than
    silently choosing an acquisition - see MultipleAcquisitionsError
    below.
    """
    try:
        volume = await get_dicom_volume(series_id, acquisition_number=acquisition_number)
    except DicomSeriesNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MultipleAcquisitionsError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(exc),
                "acquisitions": exc.acquisitions,
            },
        ) from exc
    except VolumeGeometryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "shape": list(volume.shape),
        "dtype": "int16",
        "pixel_spacing": list(volume.pixel_spacing),
        "inter_slice_spacing": volume.inter_slice_spacing,
        "orientation": volume.orientation,
        "origin": list(volume.origin),
        "ordering_method": volume.ordering_method,
        "slice_direction": volume.slice_direction,
    }


@router.get("/dicom/{series_id}/volume/data")
async def get_dicom_volume_data(series_id: str, acquisition_number: Optional[str] = None):
    """Raw binary volume data - Int16, no base64, no JSON, no temp files.

    Casts to int16 ONLY at this serialization boundary - volume.py's
    own float64 internal representation is completely unchanged.

    acquisition_number is OPTIONAL - same behavior/error contract as
    the metadata endpoint above. The returned binary corresponds ONLY
    to the selected acquisition group - build_volume() is called on
    exactly one candidate group, never a merge of several.
    """
    try:
        volume = await get_dicom_volume(series_id, acquisition_number=acquisition_number)
    except DicomSeriesNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MultipleAcquisitionsError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(exc),
                "acquisitions": exc.acquisitions,
            },
        ) from exc
    except VolumeGeometryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    int16_volume = volume.volume.astype(np.int16)
    raw_bytes = int16_volume.tobytes()

    return Response(content=raw_bytes, media_type="application/octet-stream")