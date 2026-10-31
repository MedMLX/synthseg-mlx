"""medmlx declaration and end-to-end runner for SynthSeg 2.0 and 1.0 variants.

Implements input admission, output checks and label names, and the native MLX
execution. SynthSeg's preprocessing, postprocessing, NIfTI and CSV helpers are
ported in this package (see THIRD_PARTY_NOTICES.md). The mask is returned in the
input's orientation, on the input grid or a resampled 1 mm grid; inside the networks
the volume is in RAS order.
Research software: outputs need human review and are not ground truth.
"""

from __future__ import annotations

import csv
import gzip
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from synthseg_mlx._typing import (
    DoubleArray,
    FloatArray,
    HostArray,
    ImageHeader,
    IndexArray,
    LabelArray,
    NameArray,
    NiftiApi,
    RealArray,
    Size,
)

if TYPE_CHECKING:
    from synthseg_mlx.mlx_model import SynthSegModel

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError

from synthseg_mlx.variants import (
    UPSTREAM_REVISION,
    ArithmeticProfile,
    SynthSegVariant,
    parse_variant,
)

MODEL = {
    "id": "synthseg",
    "label": "SynthSeg 2.0",
    "task": "segmentation",
    "run": "synthseg_mlx.medmlx:run",
    "summary": (
        "Brain MRI (or CT in Hounsfield units) segmentation with cortical parcellation, "
        "regional volumes and QC scores."
    ),
}

# Source revision recorded for the standard variant (the code extracted into this
# package); the other variants record UPSTREAM_REVISION. See THIRD_PARTY_NOTICES.md.
SYNTHSEG_REVISION = "d6a64db19ec7b8b71c64edd59e7e485e0235c26c"
OUTPUT_FILES = {
    "mask": "mask.nii.gz",
    "resampled_input": "resampled_input.nii.gz",
    "volumes": "volumes.csv",
    "qc_scores": "qc.csv",
    "posteriors": "posteriors.nii.gz",
}
# Written only when save_posteriors is true; always reserved against overwrite.
_OPTIONAL_OUTPUT_ROLES = frozenset({"posteriors"})
# Fixed settings, as in upstream's CLI: minimum padded size per axis (voxels), the QC
# crop size (voxels), and the sigma (voxels) of the Gaussian smoothing of posteriors.
_MIN_PAD = 128
_QC_SHAPE = 224
_SIGMA = 0.5
_WEIGHTS_HINT = (
    "Omit weights_path to download the native MedMLX/synthseg-mlx bundle automatically, "
    "or supply a folder holding the selected variant's native safetensors files."
)


class LabelFact(TypedDict):
    """A segmentation or parcellation label value and its FreeSurfer-table name."""

    value: int
    name: str


class RuntimeFacts(TypedDict):
    """Runtime record under ``details.runtime``: versions, device and stage timings.

    Timings are seconds; ``peak_memory_bytes`` is MLX's peak memory from the start of
    execution through the network pass (postprocessing is not included).
    ``source_revision`` is ``SYNTHSEG_REVISION`` for the standard variant and
    ``UPSTREAM_REVISION`` otherwise; ``volume_accumulator`` is float64 for standard.
    """

    backend: str
    mlx: str
    numpy: str
    precision: str
    arithmetic_profile: ArithmeticProfile
    volume_accumulator: str
    source_revision: str
    mlx_default_device: str
    preprocessing_seconds: float
    model_load_seconds: float
    network_seconds: float
    postprocessing_seconds: float
    peak_memory_bytes: int | None


class OutputDetails(TypedDict):
    """Facts read back from the written files by ``_check_outputs``.

    ``grid`` is ``synthseg_1mm`` when the image was resampled and ``input`` otherwise;
    ``affine`` and ``shape`` are the mask's; ``voxel_counts`` are keyed by label name.
    """

    labels: list[LabelFact]
    voxel_counts: dict[str, int]
    shape: list[int]
    dtype: str
    affine: list[list[float]]
    grid: str
    volumes_mm3: dict[str, float]
    qc_scores: dict[str, float]


class RunDetails(OutputDetails):
    """``OutputDetails`` plus the call options and runtime facts added by ``run``."""

    weights_revision: str | None
    ct: bool
    runtime: RuntimeFacts
    duration_seconds: float
    variant: str
    robust: bool
    fast: bool
    v1: bool
    parc: bool


class RunResult(TypedDict):
    """Return value of ``run``: output file paths by role, and the run details."""

    outputs: dict[str, str]
    details: RunDetails


@dataclass(frozen=True, slots=True)
class LabelSelection:
    """Label arrays for one variant, each indexed like the network channels.

    ``segmentation`` and ``parcellation`` are sorted unique label values (background
    first); ``segmentation_names``, ``topology_classes``, ``qc`` and ``qc_names`` align
    with ``segmentation``, and the parcellation names omit the background.
    ``volume_labels`` and ``volume_names`` list the ``volumes.csv`` columns (the
    background is dropped on write); total intracranial is last here and written first.
    ``flip_indices`` swaps left and right labels and is empty for fast variants.
    """

    segmentation: IndexArray
    segmentation_names: NameArray
    topology_classes: IndexArray
    parcellation: IndexArray
    volume_labels: IndexArray
    volume_names: NameArray
    qc: IndexArray
    qc_names: NameArray
    flip_indices: IndexArray


@dataclass(frozen=True, slots=True)
class _PreparedImage:
    """Outputs of ``preprocess`` plus the source header used when writing results."""

    network_input: RealArray
    affine: DoubleArray
    resolution: RealArray
    shape: list[int]
    pad_indices: IndexArray
    crop_indices: IndexArray | None
    header: ImageHeader


def run(
    *,
    image_path: str | os.PathLike[str],
    weights_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str],
    ct: bool = False,
    robust: bool = False,
    fast: bool = False,
    v1: bool = False,
    crop: int | Sequence[int] | None = None,
    parc: bool = True,
    save_posteriors: bool = False,
    overwrite: bool = False,
    arithmetic_profile: ArithmeticProfile = "source-fp32",
) -> RunResult:
    """Segment one 3D brain NIfTI with SynthSeg, optionally parcellate cortex, and score QC.

    MRI intensities are clipped at the 0.5th and 99.5th percentiles and scaled to
    [0, 1]; CT is first clipped to [0, 80] HU. The input must be finite and
    nonconstant, with at least three voxels per axis. The source dimension heuristic
    additionally rejects a final axis of three voxels. Four through ten emit a
    warning and can fail later in padding when that axis remains ten or smaller.
    Normalization does not establish model accuracy for every contrast or acquisition.
    The image is resampled to 1 mm when a voxel size differs by more than 0.05 mm and
    reoriented to RAS for the networks; the written mask is oriented like the
    (resampled) input. Labels follow FreeSurfer's lookup table.

    Args:
        image_path: One 3D brain MRI ``.nii``/``.nii.gz`` (any contrast or resolution),
            or a CT in Hounsfield units when ``ct`` is true. Spatial units must be mm
            or unspecified (assumed mm), header spacing must match the affine, and
            the grid axes must be orthogonal.
        weights_path: Optional local directory with the selected segmentation
            checkpoint and the 2.0 parcellation and QC checkpoints, even with
            ``parc=False``. All three must be native ``.safetensors`` files.
            Omit to download and reuse the MedMLX/synthseg-mlx bundle automatically.
        output_dir: Directory for the outputs; created if missing.
        ct: Clip intensities to [0, 80] before normalization, for CT input.
        robust: Use the separate 2.0 cascade; implies fast and excludes v1.
        fast: Skip left/right averaging and topological-class cleanup.
        v1: Use 1.0 weights/labels, 18 neutral labels and no intracranial volume.
        crop: Centre-crop size in voxels (one int, a one-element sequence or three ints
            in RAS order, rounded up to multiples of 32) analysed before padding, as
            upstream's ``--crop``. ``None`` (default) pads to 128 without cropping;
            otherwise this also becomes the minimum pad, matching upstream.
        parc: ``True`` (default) replaces cortex labels 3/42 with Desikan-Killiany
            parcels (1001-1035, 2001-2035) in the mask and reports their volumes, as
            upstream's ``--parc``. ``False`` keeps cortex labels 3/42 and omits parcel
            volume columns; QC uses the same segmentation output in both cases,
            which differs from upstream's ``--qc`` without ``--parc`` (see
            docs/technical.md).
        save_posteriors: ``False`` (default); ``True`` also writes ``posteriors.nii.gz``
            (upstream's ``--post``): per-voxel segmentation-label probabilities, float32,
            channels in sorted unique segmentation-label order, background first,
            on the mask's grid. ``details.labels`` also includes parcellation labels.
        overwrite: Replace existing outputs; input/output aliases are always refused.
        arithmetic_profile: ``source-fp32`` (default) keeps each variant's arithmetic.
            ``compensated-fp32`` is limited to the standard 2.0 graph and uses
            compensated Metal operators for segmentation/parcellation convolution,
            normalization, ELU, softmax and smoothing. It smooths each segmentation
            prediction before averaging. QC retains its usual arithmetic.

    Writes ``mask.nii.gz`` (int32 FreeSurfer labels, with cortex parcellated when
    ``parc`` is true),
    ``resampled_input.nii.gz``, ``volumes.csv`` (mm³), ``qc.csv`` and, when
    ``save_posteriors`` is true, ``posteriors.nii.gz``. The mask is on SynthSeg's
    input grid or a resampled 1 mm grid: inputs whose voxel sizes differ
    from 1 mm by more than 0.05 mm are resampled, and ``resampled_input.nii.gz``
    holds the image on the mask's grid (a copy of the input when no resampling was
    needed). Output requires human review and is not ground truth.

    Returns:
        ``{"outputs": {role: path}, "details": {...}}``. Roles are ``mask``,
        ``resampled_input``, ``volumes``, ``qc_scores`` and, when requested,
        ``posteriors``. ``details`` holds label facts, volumes (mm^3), QC scores, the
        variant flags and runtime timings.

    Raises:
        InvalidInputError: For invalid options, an unreadable, non-3D, non-finite or
            constant image, a degenerate affine, or existing outputs without ``overwrite``.
        AssetNotReadyError: If the weights directory or a required checkpoint is
            missing, or a checkpoint does not match the expected network.
        ModelExecutionError: If the written files fail the output checks.
    """

    started = time.perf_counter()
    variant = parse_variant(robust=robust, fast=fast, v1=v1)
    if arithmetic_profile not in {"source-fp32", "compensated-fp32"}:
        raise InvalidInputError(f"Unknown SynthSeg arithmetic profile {arithmetic_profile!r}")
    if arithmetic_profile == "compensated-fp32" and variant is not SynthSegVariant.STANDARD:
        raise InvalidInputError("Compensated SynthSeg supports the standard 2.0 graph")
    if not isinstance(cast(object, ct), bool):
        raise InvalidInputError("SynthSeg ct must be true or false")
    if not isinstance(cast(object, parc), bool):
        raise InvalidInputError("SynthSeg parc must be true or false")
    if not isinstance(cast(object, save_posteriors), bool):
        raise InvalidInputError("SynthSeg save_posteriors must be true or false")
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("SynthSeg overwrite must be true or false")
    crop = _validate_crop(crop)
    image = Path(image_path).resolve()
    if not image.is_file():
        raise InvalidInputError(f"SynthSeg input image does not exist: {image}")
    if not image.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError("SynthSeg input must be a .nii or .nii.gz file")
    from medmlx_core.weights import weights_dir, weights_revision

    models_dir = (weights_dir("synthseg") if weights_path is None
                  else Path(weights_path).expanduser())
    _require_weight_files(models_dir, variant)
    output = Path(output_dir).resolve()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"SynthSeg output_dir is not a directory: {output}")
    paths = {role: output / name for role, name in OUTPUT_FILES.items()}
    if any(
        path.resolve() == image or (path.exists() and path.samefile(image))
        for path in paths.values()
    ):
        raise InvalidInputError("SynthSeg output files must not replace the input image")
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise InvalidInputError(
            f"SynthSeg outputs already exist in {output}: {names}; "
            "pass overwrite=true to replace them"
        )
    _validate_image(image)
    # Remove stale outputs first: the CSV writer appends to the files it creates, and
    # _check_outputs treats an existing resampled_input as proof that resampling ran.
    for path in existing:
        path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    try:
        runtime = _execute(
            image,
            models_dir,
            paths,
            ct=ct,
            variant=variant,
            crop=crop,
            parc=parc,
            save_posteriors=save_posteriors,
            arithmetic_profile=arithmetic_profile,
        )
    except Exception as exc:
        # The source preprocessor reports unsupported dimensions as bare Exceptions.
        if type(exc) is Exception:
            raise InvalidInputError(f"SynthSeg cannot preprocess {image.name}: {exc}") from exc
        raise
    details = _check_outputs(image, paths, variant=variant, save_posteriors=save_posteriors)
    cast(dict[str, object], details).update(
        ct=ct, runtime=runtime, duration_seconds=round(time.perf_counter() - started, 3),
        weights_revision=weights_revision(models_dir)
    )
    cast(dict[str, object], details).update(
        variant=variant.value,
        robust=variant.robust,
        fast=variant.fast,
        v1=variant.v1,
        parc=parc,
    )
    return {
        "outputs": {
            role: path.as_posix()
            for role, path in paths.items()
            if role not in _OPTIONAL_OUTPUT_ROLES or save_posteriors
        },
        "details": cast(RunDetails, details),
    }


def _validate_crop(crop: int | Sequence[int] | None) -> int | list[int] | None:
    """Admission check for upstream's ``--crop``: one int, or one per axis."""

    if crop is None:
        return None
    message = "SynthSeg crop must be an int, a sequence of ints, or null"
    value = cast(object, crop)
    if isinstance(value, bool):
        raise InvalidInputError(message)
    if isinstance(value, int):
        return value
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise InvalidInputError(message)
    items = list(cast("Sequence[object]", value))
    if not items or any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise InvalidInputError(message)
    return cast(list[int], items)


def _require_weight_files(models_dir: Path, variant: SynthSegVariant) -> None:
    """Raise ``AssetNotReadyError`` unless the directory holds the variant's checkpoints."""

    from synthseg_mlx.pinned_weights import checkpoint_paths

    if not models_dir.is_dir():
        raise AssetNotReadyError(
            f"SynthSeg weights directory not found: {models_dir}; expected a directory holding "
            + ", ".join(variant.checkpoints),
            reason=f"missing {models_dir}",
            hint=_WEIGHTS_HINT,
        )
    try:
        checkpoint_paths(models_dir, variant)
    except ValueError as exc:
        raise AssetNotReadyError(
            f"SynthSeg weights directory {models_dir}: {exc}",
            reason=str(exc),
            hint=_WEIGHTS_HINT,
        ) from exc


def _validate_image(image: Path) -> None:
    """Check image values and the millimetre geometry used for processing and volumes."""

    import nibabel as nib_runtime

    nib = cast(NiftiApi, nib_runtime)

    try:
        loaded = nib.load(str(image))
        data = cast(RealArray, np.asanyarray(loaded.dataobj))
    except Exception as exc:
        raise InvalidInputError(f"SynthSeg could not read {image}: {exc}") from exc
    if len(loaded.shape) != 3 or min(loaded.shape) < 3:
        raise InvalidInputError("SynthSeg requires one 3D brain volume")
    if not np.isfinite(data).all() or data.max() <= data.min():
        raise InvalidInputError("SynthSeg input must contain finite, nonconstant intensities")
    if not np.isfinite(loaded.affine).all() or abs(np.linalg.det(loaded.affine[:3, :3])) < 1e-8:
        raise InvalidInputError("SynthSeg input has invalid physical geometry")
    try:
        spatial_units, _ = loaded.header.get_xyzt_units()
    except (AttributeError, KeyError, ValueError) as exc:
        raise InvalidInputError("SynthSeg input has invalid NIfTI spatial units") from exc
    if spatial_units not in ("mm", "unknown"):
        raise InvalidInputError("SynthSeg requires NIfTI spatial units in mm or unspecified")
    # Upstream reads pixdim for resampling but uses the selected affine for output
    # geometry. A disagreement would assign the wrong physical scale to the result.
    header_spacing = np.asarray(loaded.header["pixdim"][1:4], dtype=np.float64)
    affine_spacing = np.linalg.norm(loaded.affine[:3, :3], axis=0)
    if not np.allclose(header_spacing, affine_spacing, rtol=1e-5, atol=1e-6):
        raise InvalidInputError("SynthSeg NIfTI voxel spacing must match its selected affine")
    # Source volume reporting multiplies axis lengths, which is only voxel volume
    # for orthogonal grids. Rotated axes are supported; sheared axes are not.
    directions = loaded.affine[:3, :3] / affine_spacing
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=0.0, atol=1e-5):
        raise InvalidInputError("SynthSeg does not support sheared NIfTI geometry")


def _execute(
    image: Path,
    models_dir: Path,
    paths: dict[str, Path],
    *,
    ct: bool,
    variant: SynthSegVariant,
    crop: Size | None = None,
    parc: bool = True,
    save_posteriors: bool = False,
    arithmetic_profile: ArithmeticProfile = "source-fp32",
) -> RuntimeFacts:
    """Run the selected segmentation, parcellation and QC graphs natively in MLX."""

    from medmlx_core.runtime import (
        import_mlx,
        mlx_default_device_name,
    )

    mlxc = import_mlx()
    from synthseg_mlx.mlx_engine import predict_numpy

    labels = select_labels(variant, parc=parc)
    mlxc.reset_peak_memory()
    prepared_started = time.perf_counter()
    prepared = _prepare_image(image, paths["resampled_input"], ct=ct, crop=crop)
    preprocessing_seconds = time.perf_counter() - prepared_started
    loaded_started = time.perf_counter()
    model = _load_model(models_dir, labels, variant, arithmetic_profile=arithmetic_profile)
    model_load_seconds = time.perf_counter() - loaded_started
    _write_tables(paths, labels, variant)
    network_started = time.perf_counter()
    predictions = predict_numpy(model, prepared.network_input)
    network_seconds = time.perf_counter() - network_started
    peak_memory_bytes = mlxc.get_peak_memory()
    postprocess_started = time.perf_counter()
    posteriors_path = paths["posteriors"] if save_posteriors else None
    _write_prediction(image, paths, prepared, labels, predictions, variant, parc, posteriors_path)
    postprocessing_seconds = time.perf_counter() - postprocess_started
    return {
        "backend": "mlx",
        "mlx": version("mlx"),
        "numpy": np.__version__,
        "precision": "float32",
        "arithmetic_profile": arithmetic_profile,
        "volume_accumulator": "float64" if variant is SynthSegVariant.STANDARD else "float32",
        "source_revision": (
            SYNTHSEG_REVISION if variant is SynthSegVariant.STANDARD else UPSTREAM_REVISION
        ),
        "mlx_default_device": mlx_default_device_name(mlxc),
        "preprocessing_seconds": round(preprocessing_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "network_seconds": round(network_seconds, 6),
        "postprocessing_seconds": round(postprocessing_seconds, 6),
        "peak_memory_bytes": peak_memory_bytes,
    }


def _prepare_image(
    image: Path, resampled_path: Path, *, ct: bool, crop: Size | None = None
) -> _PreparedImage:
    """Load ``image`` and preprocess it; a resampled copy, if made, goes to ``resampled_path``."""

    from synthseg_mlx._lab2im import utils
    from synthseg_mlx.processing import preprocess

    loaded, _, affine, _, _, header, resolution = utils.get_volume_info(str(image), True)

    def save_resampled(resampled: RealArray, resampled_affine: DoubleArray) -> None:
        """Write the prepared image and affine when resampling occurred."""
        utils.save_volume(resampled, resampled_affine, header, str(resampled_path))

    # Upstream's CLI sets min_pad to the crop size when --crop is given, and to
    # 128 otherwise (SynthSeg_predict.py / predict_synthseg.py).
    network_input, affine, resolution, shape, pad_idx, crop_idx = preprocess(
        loaded,
        affine,
        resolution,
        ct=ct,
        crop=crop,
        min_pad=crop if crop is not None else _MIN_PAD,
        on_resample=save_resampled,
    )
    return _PreparedImage(network_input, affine, resolution, shape, pad_idx, crop_idx, header)


def _load_model(
    models_dir: Path,
    labels: LabelSelection,
    variant: SynthSegVariant,
    *,
    arithmetic_profile: ArithmeticProfile = "source-fp32",
) -> SynthSegModel:
    """Build the model from the variant's checkpoints; bad weights raise ``AssetNotReadyError``."""

    from synthseg_mlx.mlx_model import SynthSegModel
    from synthseg_mlx.pinned_weights import checkpoint_paths

    options = dict(robust=variant.robust, fast=variant.fast, v1=variant.v1)
    try:
        models = checkpoint_paths(models_dir, variant)
    except ValueError as exc:
        raise AssetNotReadyError(str(exc), reason=str(exc), hint=_WEIGHTS_HINT) from exc
    try:
        return SynthSegModel.from_safetensors(
            models[variant.segmentation_checkpoint],
            labels.segmentation,
            models["synthseg_parc_2.0.safetensors"],
            labels.parcellation,
            models["synthseg_qc_2.0.safetensors"],
            labels.qc,
            labels.flip_indices,
            sigma=_SIGMA,
            qc_shape=_QC_SHAPE,
            labels_denoiser=denoiser_labels() if variant.robust else None,
            arithmetic_profile=arithmetic_profile,
            **options,
        )
    except (ModelExecutionError, OSError) as exc:
        raise AssetNotReadyError(
            f"SynthSeg weights in {models_dir} do not match the "
            f"{'1.0' if variant.v1 else '2.0'} networks: {exc}",
            reason=str(exc),
            hint=_WEIGHTS_HINT,
        ) from exc


def _write_tables(
    paths: dict[str, Path],
    labels: LabelSelection,
    variant: SynthSegVariant,
    *,
    volumes_row: list[str] | None = None,
    qc_row: list[str] | None = None,
) -> None:
    """Write ``volumes.csv`` and ``qc.csv`` with SynthSeg's header and one subject row."""

    from synthseg_mlx.labels import write_csv

    # Initialize the source's subject/header contract, then append one result row.
    write_csv(
        str(paths["volumes"]),
        volumes_row,
        True,
        labels.volume_labels,
        labels.volume_names,
        last_first=not variant.v1,
    )
    write_csv(str(paths["qc_scores"]), qc_row, True, labels.qc, labels.qc_names)


def _write_prediction(
    image: Path,
    paths: dict[str, Path],
    prepared: _PreparedImage,
    labels: LabelSelection,
    predictions: tuple[FloatArray, FloatArray, FloatArray],
    variant: SynthSegVariant,
    parc: bool = True,
    posteriors_path: Path | None = None,
) -> None:
    """Postprocess the network outputs and write the mask, optional posteriors and tables."""

    from synthseg_mlx._lab2im import utils
    from synthseg_mlx.mlx_engine import postprocess_arrays, qc_scores

    options = dict(robust=variant.robust, fast=variant.fast, v1=variant.v1)
    segmentation, parcellation, quality = predictions
    mask, posteriors, volumes = postprocess_arrays(
        segmentation,
        parcellation if parc else None,
        shape=prepared.shape,
        pad_idx=prepared.pad_indices,
        crop_idx=prepared.crop_indices,
        labels_segmentation=labels.segmentation,
        labels_parcellation=labels.parcellation,
        affine=prepared.affine,
        image_resolution=prepared.resolution,
        topology_classes=labels.topology_classes,
        **options,
    )
    utils.save_volume(mask, prepared.affine, prepared.header, str(paths["mask"]), dtype="int32")
    if posteriors_path is not None:
        utils.save_volume(
            cast(RealArray, posteriors),
            prepared.affine,
            prepared.header,
            str(posteriors_path),
            dtype="float32",
        )
    subject = image.name.replace(".nii.gz", "")
    _write_tables(
        paths,
        labels,
        variant,
        volumes_row=[subject, *(str(value) for value in volumes)],
        qc_row=[subject, *(f"{value:.4f}" for value in qc_scores(quality, **options))],
    )


def select_labels(
    variant: SynthSegVariant = SynthSegVariant.STANDARD, *, parc: bool = True
) -> LabelSelection:
    """Select the bundled label arrays for ``variant``, in network channel order.

    Non-fast variants (standard, v1) use ``get_flip_indices`` for sorted labels plus
    left/right flip indices; fast variants (fast, robust, v1_fast) only sort. ``parc``
    adds the parcellation labels to the volume columns; the total intracranial column
    (label ``max + 1``) is added unless v1.
    """

    from synthseg_mlx._lab2im import utils
    from synthseg_mlx.label_data import load_label_data
    from synthseg_mlx.labels import get_flip_indices

    data = load_label_data(v1=variant.v1)
    segmentation, _ = utils.get_list_labels(label_list=data["segmentation"])
    if variant.fast:
        segmentation, unique_indices = np.unique(segmentation, return_index=True)
        flip_indices = np.array([], dtype=np.int32)
    else:
        segmentation, flip_indices, unique_indices = get_flip_indices(
            segmentation,
            variant.neutral_labels,
        )
    segmentation_names = utils.load_array_if_path(data["segmentation_names"])[unique_indices]
    topology_classes = utils.load_array_if_path(data["topology_classes"], load_as_numpy=True)[
        unique_indices
    ]
    parcellation, parcellation_indices = np.unique(
        utils.get_list_labels(label_list=data["parcellation"])[0],
        return_index=True,
    )
    parcellation_names = utils.load_array_if_path(data["parcellation_names"])[parcellation_indices][
        1:
    ]
    if parc:
        volume_labels = np.concatenate((segmentation, parcellation[1:]))
        volume_names = np.concatenate((segmentation_names, parcellation_names))
    else:
        volume_labels = np.asarray(segmentation)
        volume_names = np.asarray(segmentation_names)
    if not variant.v1:
        volume_labels = np.concatenate((volume_labels, np.array([np.max(volume_labels + 1)])))
        volume_names = np.concatenate((volume_names, np.array(["total intracranial"])))
    qc = utils.get_list_labels(label_list=data["qc"])[0][unique_indices]
    qc_names = utils.load_array_if_path(data["qc_names"])[unique_indices]
    return LabelSelection(
        segmentation=np.asarray(segmentation),
        segmentation_names=np.asarray(segmentation_names),
        topology_classes=np.asarray(topology_classes),
        parcellation=np.asarray(parcellation),
        volume_labels=np.asarray(volume_labels),
        volume_names=np.asarray(volume_names),
        qc=np.asarray(qc),
        qc_names=np.asarray(qc_names),
        flip_indices=np.asarray(flip_indices),
    )


def denoiser_labels() -> LabelArray:
    """Tissue-group labels (0-4) of the robust denoiser, sorted and unique."""
    from synthseg_mlx.label_data import load_label_data

    return np.unique(load_label_data(robust=True)["denoiser"])


def label_names(*, v1: bool = False) -> dict[int, str]:
    """Map every segmentation and parcellation label value to its name."""

    from synthseg_mlx.label_data import load_label_data

    data = load_label_data(v1=v1)
    names: dict[int, str] = {}
    for prefix in ("segmentation", "parcellation"):
        values = cast(IndexArray, cast(Mapping[str, HostArray], data)[prefix])
        labels = cast(NameArray, cast(Mapping[str, HostArray], data)[f"{prefix}_names"])
        names.update({int(value): str(name) for value, name in zip(values, labels, strict=True)})
    return names


def _check_outputs(
    image: Path,
    paths: dict[str, Path],
    *,
    variant: SynthSegVariant = SynthSegVariant.STANDARD,
    save_posteriors: bool = False,
) -> OutputDetails:
    """Validate the written files and collect the ``details`` facts (labels, grid, tables)."""

    import nibabel as nib_runtime

    nib = cast(NiftiApi, nib_runtime)

    if not paths["mask"].is_file():
        raise ModelExecutionError("SynthSeg did not write a mask")
    mask = nib.load(str(paths["mask"]))
    labels = cast(RealArray, np.asanyarray(mask.dataobj))
    if (
        not np.isfinite(labels).all()
        or not np.any(labels)
        or not np.equal(labels, np.round(labels)).all()
    ):
        raise ModelExecutionError("SynthSeg wrote an empty or invalid label map")
    names = label_names(v1=variant.v1)
    if not set(np.unique(labels)).issubset(names):
        raise ModelExecutionError("SynthSeg wrote labels outside its declared parcellation")
    resampled = paths["resampled_input"].is_file()
    if not resampled:
        source = nib.load(str(image))
        if source.shape != mask.shape or not np.allclose(source.affine, mask.affine, atol=1e-4):
            raise ModelExecutionError("SynthSeg did not write the matching resampled image")
        # Upstream omits this image when resampling is unnecessary.
        copy_nifti(image, paths["resampled_input"])
    reference = nib.load(str(paths["resampled_input"]))
    if reference.shape != mask.shape or not np.allclose(reference.affine, mask.affine, atol=1e-4):
        raise ModelExecutionError("SynthSeg mask and resampled image have different grids")
    if save_posteriors:
        if not paths["posteriors"].is_file():
            raise ModelExecutionError("SynthSeg did not write posteriors")
        posteriors_image = nib.load(str(paths["posteriors"]))
        posteriors = cast(RealArray, np.asanyarray(posteriors_image.dataobj))
        if (
            posteriors.shape[:3] != mask.shape
            or posteriors.ndim != 4
            or posteriors_image.get_data_dtype() != np.dtype("float32")
            or not np.allclose(posteriors_image.affine, mask.affine, atol=1e-4)
            or not np.isfinite(posteriors).all()
            or not np.allclose(np.sum(posteriors, axis=-1), 1.0, atol=1e-3)
        ):
            raise ModelExecutionError("SynthSeg wrote invalid posteriors")
    tables: dict[str, dict[str, float]] = {}
    for role in ("volumes", "qc_scores"):
        with paths[role].open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        if len(rows) != 2 or len(rows[0]) != len(rows[1]):
            raise ModelExecutionError(f"SynthSeg did not write one complete {role} row")
        if not np.isfinite(np.asarray(rows[1][1:], dtype=float)).all():
            raise ModelExecutionError(f"SynthSeg wrote invalid {role}")
        tables[role] = {
            name: float(value) for name, value in zip(rows[0][1:], rows[1][1:], strict=True)
        }
    values, counts = np.unique(labels, return_counts=True)
    return {
        "labels": [{"value": value, "name": name} for value, name in sorted(names.items())],
        "voxel_counts": {
            names[int(value)]: int(count) for value, count in zip(values, counts, strict=True)
        },
        "shape": [int(size) for size in mask.shape],
        "dtype": str(mask.get_data_dtype()),
        "affine": cast(list[list[float]], np.asarray(mask.affine, dtype=float).tolist()),
        "grid": "synthseg_1mm" if resampled else "input",
        "volumes_mm3": tables["volumes"],
        "qc_scores": tables["qc_scores"],
    }


def copy_nifti(source: Path, destination: Path) -> None:
    """Copy a NIfTI file to a ``.nii.gz`` destination, gzipping a plain ``.nii`` source.

    When compressing a plain ``.nii``, the gzip header carries no file name or
    timestamp. A ``.nii.gz`` source is copied unchanged, including its header.
    """

    if source.name.endswith(".nii.gz"):
        shutil.copy2(source, destination)
        return
    with (
        source.open("rb") as input_file,
        destination.open("wb") as raw_output,
        gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=0) as output,
    ):
        shutil.copyfileobj(input_file, output)


__all__ = ["MODEL", "OUTPUT_FILES", "SYNTHSEG_REVISION", "label_names", "run"]
