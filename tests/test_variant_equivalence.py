"""Reduced seeded graphs and complete runners against upstream TensorFlow/Keras."""

import json
from pathlib import Path
from typing import Protocol, TypedDict, Unpack, cast

import mlx.core as mx
import nibabel as nib_runtime
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, ModelExecutionError
from recording_types import Recording
from variant_fixtures import (
    assert_reporting_units,
    load_standard_volume_reference,
    load_variant,
    write_weights,
)

from synthseg_mlx import (
    SynthSegModel,
    medmlx,
    postprocess_arrays,
    predict_numpy,
    processing,
    qc_scores,
)
from synthseg_mlx._typing import (
    DoubleArray,
    FloatArray,
    IndexArray,
    NativeArrayApi,
    NiftiApi,
    RealArray,
    Size,
)
from synthseg_mlx.variants import SynthSegVariant


class Flags(TypedDict):
    robust: bool
    fast: bool
    v1: bool


class PreprocessOptions(TypedDict):
    ct: bool
    crop: Size | None
    min_pad: Size | None


class LoaderOptions(Flags):
    sigma: float
    qc_shape: int
    labels_denoiser: IndexArray | None


class SafeTensorWriter(Protocol):
    def save_safetensors(self, file: str, arrays: dict[str, mx.array]) -> None:
        """Write named native arrays to a safetensors file."""
        ...


nib = cast(NiftiApi, nib_runtime)


type ModelCase = tuple[Recording, SynthSegModel, list[Path], Flags]


def model_case(directory: Path, name: str) -> tuple[Recording, SynthSegModel, list[Path], Flags]:
    """Construct a reduced fixture model and checkpoint paths for a variant."""
    fixture = load_variant(name)
    paths = write_weights(directory, fixture)
    options = cast(Flags, {key: bool(fixture[key]) for key in ("robust", "fast", "v1")})
    model = SynthSegModel.from_safetensors(
        paths[0],
        fixture["labels_segmentation"],
        paths[1],
        fixture["labels_parcellation"],
        paths[2],
        fixture["labels_qc"],
        fixture["flip_indices"],
        labels_denoiser=fixture["labels_denoiser"],
        features=int(fixture["features"]),
        denoiser_features=int(fixture["denoiser_features"]),
        qc_shape=int(fixture["qc_shape"]),
        **options,
    )
    return fixture, model, paths, options


@pytest.fixture(params=["standard", "fast", "robust", "v1", "v1_fast"])
def variant_case(request: pytest.FixtureRequest, tmp_path: Path) -> ModelCase:
    """Provide each source variant with its recorded data and native graph."""
    return model_case(tmp_path, cast(str, request.param))


def test_safetensors_graph_matches_tensorflow_with_cache_symlinks(
    variant_case: ModelCase, tmp_path: Path
) -> None:
    """Compare restored symlinked checkpoints and source graph outputs."""
    fixture, original, _, options = variant_case
    paths: list[Path] = []
    for name, component in zip(
        ("segmentation", "parcellation", "qc"),
        (original.segmentation, original.parcellation, original.qc),
        strict=True,
    ):
        path = tmp_path / f"{name}.safetensors"
        component.save_weights(str(path))
        blob = tmp_path / f"{name}-blob"
        path.rename(blob)
        path.symlink_to(blob)
        paths.append(path)
    restored = SynthSegModel(
        fixture["labels_segmentation"],
        fixture["labels_parcellation"],
        fixture["labels_qc"],
        fixture["flip_indices"],
        labels_denoiser=fixture["labels_denoiser"],
        features=int(fixture["features"]),
        denoiser_features=int(fixture["denoiser_features"]),
        qc_shape=int(fixture["qc_shape"]),
        **options,
    )
    restored.load_safetensors(paths[0], paths[1], paths[2])
    for field, actual in zip(
        ("segmentation", "parcellation", "qc"),
        predict_numpy(restored, fixture["input"]),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, cast(RealArray, fixture[field]), rtol=1e-5, atol=1e-6, err_msg=field
        )


@pytest.mark.parametrize("invalid", ["dtype", "nonfinite", "missing", "shape"])
def test_safetensors_reject_invalid_parameters(tmp_path: Path, invalid: str) -> None:
    """Reject missing, malformed, or non-finite checkpoint tensors."""
    _, model, _, _ = model_case(tmp_path, "standard")
    seg, parc, qc = (tmp_path / f"{name}.safetensors" for name in ("seg", "parc", "qc"))
    model.segmentation.save_weights(str(seg))
    model.parcellation.save_weights(str(parc))
    model.qc.save_weights(str(qc))
    weights = cast("dict[str, mx.array]", mx.load(str(seg), format="safetensors"))
    key = next(iter(weights))
    if invalid == "dtype":
        weights[key] = weights[key].astype(mx.float16)
    elif invalid == "nonfinite":
        weights[key] = mx.full(weights[key].shape, float("nan"), dtype=mx.float32)
    elif invalid == "missing":
        del weights[key]
    else:
        weights[key] = mx.zeros((1,), dtype=mx.float32)
    bad = tmp_path / "bad.safetensors"
    cast(SafeTensorWriter, mx).save_safetensors(str(bad), weights)
    with pytest.raises(ModelExecutionError):
        model.load_safetensors(bad, parc, qc)


def test_variant_graph_matches_tensorflow(variant_case: ModelCase) -> None:
    """Compare each native variant graph and its outputs with TensorFlow recordings."""
    fixture, model, _, options = variant_case
    predictions = predict_numpy(model, fixture["input"])
    whole_graph = model(cast(NativeArrayApi, mx).array(fixture["input"]))
    cast(NativeArrayApi, mx).eval(*whole_graph)
    for field, actual, unstaged in zip(
        ("segmentation", "parcellation", "qc"), predictions, whole_graph, strict=True
    ):
        assert actual.dtype == fixture[field].dtype
        np.testing.assert_allclose(
            actual, cast(RealArray, fixture[field]), rtol=1e-5, atol=1e-6, err_msg=field
        )
        # Stage-by-stage evaluation must not change any value of the one-graph result.
        np.testing.assert_array_equal(actual, cast(RealArray, np.array(unstaged)), err_msg=field)
    _, conditioning = model.segmentation_predictions(
        cast(NativeArrayApi, mx).array(fixture["input"])
    )
    actual_cortex = model.cortex_labels[mx.argmax(conditioning, axis=-1)]
    np.testing.assert_array_equal(
        cast(RealArray, np.array(actual_cortex)),
        fixture["parcellation_input"][..., 2] > 0,
    )
    quality = qc_scores(predictions[2], **options)
    expected_dtype = np.float64 if model.variant is SynthSegVariant.STANDARD else np.float32
    assert quality.dtype == expected_dtype
    assert_reporting_units(
        quality,
        np.around(np.clip(fixture["qc"].squeeze()[1:], 0, 1), 4),
        decimals=4,
    )
    mask, _, volumes = postprocess_arrays(
        *predictions[:2],
        shape=[16] * 3,
        pad_idx=np.array([0, 0, 0, 16, 16, 16]),
        crop_idx=None,
        labels_segmentation=fixture["labels_segmentation"],
        labels_parcellation=fixture["labels_parcellation"],
        affine=np.eye(4),
        image_resolution=np.ones(3),
        topology_classes=cast(IndexArray, fixture["topology_classes"]),
        **options,
    )
    np.testing.assert_array_equal(mask, fixture["mask"])
    reference_volumes = (
        load_standard_volume_reference()
        if model.variant is SynthSegVariant.STANDARD
        else fixture["volumes"]
    )
    assert_reporting_units(volumes, reference_volumes, decimals=3)


def reduced_runner(
    fixture: Recording,
    model: SynthSegModel,
    weights: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure the runner to use the reduced recorded graph and labels."""
    original_preprocess, original_loader = processing.preprocess, SynthSegModel.from_safetensors

    def small_preprocess(
        *args: *tuple[RealArray, DoubleArray, RealArray], **kwargs: Unpack[PreprocessOptions]
    ) -> tuple[RealArray, DoubleArray, RealArray, list[int], IndexArray, IndexArray | None]:
        """Run source-order preprocessing at the reduced fixture size."""
        return original_preprocess(*args, n_levels=4, **kwargs)

    def small_model(
        *args: *tuple[Path, IndexArray, Path, IndexArray, Path, IndexArray, IndexArray | None],
        **kwargs: Unpack[LoaderOptions],
    ) -> SynthSegModel:
        """Load the selected model using reduced fixture feature counts."""
        return original_loader(*args, features=2, denoiser_features=1, **kwargs)

    # Only sizes and synthetic checkpoint pins change; every pipeline stage runs.
    monkeypatch.setattr(processing, "preprocess", small_preprocess)
    monkeypatch.setattr(medmlx, "_MIN_PAD", 16)
    monkeypatch.setattr(medmlx, "_QC_SHAPE", 16)
    monkeypatch.setattr(SynthSegModel, "from_safetensors", small_model)
    labels = medmlx.select_labels(model.variant)
    reduced = labels.__class__(
        segmentation=labels.segmentation,
        segmentation_names=labels.segmentation_names,
        topology_classes=labels.topology_classes,
        parcellation=fixture["labels_parcellation"],
        volume_labels=fixture["volume_labels"],
        volume_names=fixture["volume_names"],
        qc=labels.qc,
        qc_names=labels.qc_names,
        flip_indices=labels.flip_indices,
    )

    def reduced_labels(_variant: SynthSegVariant, *, parc: bool = True) -> medmlx.LabelSelection:
        """Return selected reduced label metadata for the fixture variant."""
        if parc:
            return reduced
        n_seg = len(fixture["labels_segmentation"])
        volume_labels, volume_names = reduced.volume_labels, reduced.volume_names
        if bool(fixture["v1"]):
            volume_labels, volume_names = volume_labels[:n_seg], volume_names[:n_seg]
        else:
            volume_labels = np.concatenate([volume_labels[:n_seg], volume_labels[-1:]])
            volume_names = np.concatenate([volume_names[:n_seg], volume_names[-1:]])
        return medmlx.LabelSelection(
            segmentation=reduced.segmentation,
            segmentation_names=reduced.segmentation_names,
            topology_classes=reduced.topology_classes,
            parcellation=reduced.parcellation,
            volume_labels=volume_labels,
            volume_names=volume_names,
            qc=reduced.qc,
            qc_names=reduced.qc_names,
            flip_indices=reduced.flip_indices,
        )

    monkeypatch.setattr(medmlx, "select_labels", reduced_labels)


def assert_runner_files(
    result: medmlx.RunResult,
    fixture: Recording,
    *,
    ct: bool = False,
    prefix: str | None = None,
    save_posteriors: bool = False,
) -> None:
    """Check runner files, geometry, labels, and reported values against a fixture."""
    json.dumps(result, allow_nan=False)
    assert set(result) == {"outputs", "details"}
    expected_roles = {"mask", "resampled_input", "volumes", "qc_scores"}
    if save_posteriors:
        expected_roles.add("posteriors")
    assert set(result["outputs"]) == expected_roles
    assert all(
        Path(path).is_absolute() and Path(path).is_file() for path in result["outputs"].values()
    )
    prefix = ("ct_" if ct else "") if prefix is None else prefix
    mask = nib.load(result["outputs"]["mask"])
    resampled = nib.load(result["outputs"]["resampled_input"])
    data = cast(IndexArray, np.asanyarray(mask.dataobj))
    np.testing.assert_array_equal(data, cast(IndexArray, fixture[f"{prefix}runner_mask"]))
    np.testing.assert_allclose(
        mask.affine, cast(DoubleArray, fixture[f"{prefix}runner_affine"]), rtol=1e-6, atol=1e-6
    )
    assert mask.get_data_dtype() == np.dtype("int32")
    assert mask.shape == resampled.shape
    np.testing.assert_allclose(resampled.affine, mask.affine, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        cast(RealArray, np.asanyarray(resampled.dataobj)),
        cast(FloatArray, fixture[f"{prefix}runner_resampled"]),
        rtol=1e-6,
        atol=1e-5,
    )
    details = result["details"]
    assert details["shape"] == list(mask.shape)
    assert details["dtype"] == "int32"
    assert details["affine"] == mask.affine.tolist()
    assert details["grid"] == ("synthseg_1mm" if ct else "input")
    assert details["ct"] is ct
    names = {entry["value"]: entry["name"] for entry in details["labels"]}
    assert set(np.unique(data)) <= names.keys()
    counts = {
        names[int(value)]: int(cast(np.int64, np.sum(data == value))) for value in np.unique(data)
    }
    assert details["voxel_counts"] == counts
    assert details["runtime"]["backend"] == "mlx"
    assert ("total intracranial" in details["volumes_mm3"]) is not bool(fixture["v1"])


def test_variant_runner_uses_the_same_graph_and_processing(
    variant_case: ModelCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare end-to-end variant runner outputs with the source fixture."""
    fixture, model, weights, options = variant_case
    reduced_runner(fixture, model, weights, monkeypatch)
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(fixture["runner_input"], fixture["runner_input_affine"]), str(image))
    result = medmlx.run(
        image_path=image,
        weights_path=tmp_path,
        output_dir=tmp_path / "out",
        **options,
    )
    assert_runner_files(result, fixture)
    assert result["details"]["variant"] == model.variant.value
    assert result["details"]["fast"] is model.variant.fast
    assert_reporting_units(
        list(result["details"]["volumes_mm3"].values()), fixture["runner_volumes"], decimals=3
    )
    quality = np.around(np.clip(fixture["runner_qc"].squeeze()[1:], 0, 1), 4)
    assert_reporting_units(list(result["details"]["qc_scores"].values()), quality, decimals=4)


def test_variant_runner_supports_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``crop`` matches upstream's ``--crop``: it also becomes the minimum pad."""

    fixture, model, weights, options = model_case(tmp_path, "fast")
    reduced_runner(fixture, model, weights, monkeypatch)
    image = tmp_path / "image.nii.gz"
    nib.save(
        nib.Nifti1Image(fixture["crop_runner_input"], fixture["crop_runner_input_affine"]),
        str(image),
    )
    result = medmlx.run(
        image_path=image,
        weights_path=tmp_path,
        output_dir=tmp_path / "out",
        crop=16,
        **options,
    )
    assert_runner_files(result, fixture, prefix="crop_")
    assert_reporting_units(
        list(result["details"]["volumes_mm3"].values()), fixture["crop_runner_volumes"], decimals=3
    )


def test_variant_runner_supports_parc_off(
    variant_case: ModelCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``parc=False`` matches upstream's ``--parc`` omitted: everything but the cortex

    substitution and its volume columns is unaffected (confirmed against the
    pinned upstream graph: the returned segmentation does not depend on
    ``do_parcellation``).
    """

    fixture, model, weights, options = variant_case
    reduced_runner(fixture, model, weights, monkeypatch)
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(fixture["runner_input"], fixture["runner_input_affine"]), str(image))
    with_parc = medmlx.run(
        image_path=image, weights_path=tmp_path, output_dir=tmp_path / "with_parc", **options
    )
    without_parc = medmlx.run(
        image_path=image,
        weights_path=tmp_path,
        output_dir=tmp_path / "without_parc",
        parc=False,
        **options,
    )
    assert with_parc["details"]["parc"] is True
    assert without_parc["details"]["parc"] is False
    assert set(without_parc["outputs"]) == set(with_parc["outputs"])

    with_mask = cast(IndexArray, np.asanyarray(nib.load(with_parc["outputs"]["mask"]).dataobj))
    without_mask = cast(
        IndexArray, np.asanyarray(nib.load(without_parc["outputs"]["mask"]).dataobj)
    )
    parcellation_labels = fixture["labels_parcellation"][1:]
    differing = with_mask != without_mask
    assert np.any(differing)
    # --parc only ever substitutes cortex labels 3/42 for parcellation codes; every
    # other voxel is unaffected by the flag.
    assert np.all(np.isin(with_mask[differing], parcellation_labels))
    assert np.all(np.isin(without_mask[differing], [3, 42]))
    assert not np.any(np.isin(without_mask, parcellation_labels))

    with_volumes, without_volumes = (
        with_parc["details"]["volumes_mm3"],
        without_parc["details"]["volumes_mm3"],
    )
    assert set(without_volumes) < set(with_volumes)
    for name, value in without_volumes.items():
        assert with_volumes[name] == value
    # QC always scores the parcellation-conditioned graph (docs/technical.md); the
    # toggle only changes whether the mask/volumes merge that branch's result.
    assert with_parc["details"]["qc_scores"] == without_parc["details"]["qc_scores"]


def test_variant_runner_supports_save_posteriors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``save_posteriors=True`` matches upstream's ``--post``: float32 probabilities on

    the mask's grid whose channel-wise argmax reproduces the written hard labels.
    """

    fixture, model, weights, options = model_case(tmp_path, "fast")
    reduced_runner(fixture, model, weights, monkeypatch)
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(fixture["runner_input"], fixture["runner_input_affine"]), str(image))
    result = medmlx.run(
        image_path=image,
        weights_path=tmp_path,
        output_dir=tmp_path / "out",
        save_posteriors=True,
        parc=False,
        **options,
    )
    assert set(result["outputs"]) == {
        "mask",
        "resampled_input",
        "volumes",
        "qc_scores",
        "posteriors",
    }
    mask_image = nib.load(result["outputs"]["mask"])
    mask = cast(IndexArray, np.asanyarray(mask_image.dataobj))
    posteriors_image = nib.load(result["outputs"]["posteriors"])
    posteriors = cast(RealArray, np.asanyarray(posteriors_image.dataobj))
    assert posteriors_image.get_data_dtype() == np.dtype("float32")
    n_labels = len(fixture["labels_segmentation"])
    assert posteriors.shape == (*mask_image.shape, n_labels)
    np.testing.assert_allclose(posteriors_image.affine, mask_image.affine, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.sum(posteriors, axis=-1), 1.0, atol=1e-5)
    # Upstream's posteriors are exactly the probabilities the hard labels argmax from.
    hard = fixture["labels_segmentation"][np.argmax(posteriors, axis=-1)]
    np.testing.assert_array_equal(hard, mask)


def test_runner_resamples_ct_and_replaces_stale_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check CT resampling and authorized replacement of existing outputs."""
    fixture, model, weights, options = model_case(tmp_path, "fast")
    reduced_runner(fixture, model, weights, monkeypatch)
    image, output = tmp_path / "image.nii", tmp_path / "out"
    nib.save(
        nib.Nifti1Image(fixture["ct_runner_input"], fixture["ct_runner_input_affine"]), str(image)
    )
    output.mkdir()
    for filename in medmlx.OUTPUT_FILES.values():
        (output / filename).write_text("stale")
    result = medmlx.run(
        image_path=image,
        weights_path=tmp_path,
        output_dir=output,
        ct=True,
        overwrite=True,
        **options,
    )
    assert_runner_files(result, fixture, ct=True)
    assert tuple(result["details"]["shape"]) != fixture["ct_runner_input"].shape


def test_run_rejects_pinned_files_without_the_networks(
    variant_case: ModelCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject checkpoint files whose tensors do not match the selected networks."""
    fixture, model, weights, options = variant_case
    unrelated = tmp_path / "unrelated.safetensors"
    cast(SafeTensorWriter, mx).save_safetensors(str(unrelated), {"unrelated": mx.zeros((1,))})
    unrelated.replace(weights[0])
    reduced_runner(fixture, model, weights, monkeypatch)
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(fixture["runner_input"], np.eye(4)), str(image))
    message = r"do not match the .* networks"
    with pytest.raises(AssetNotReadyError, match=message):
        medmlx.run(image_path=image, weights_path=tmp_path, output_dir=tmp_path / "out", **options)
