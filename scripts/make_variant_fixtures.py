"""Record reduced-width SynthSeg variants with the pinned upstream TF/Keras graph.

The recorder never imports the MLX port. It executes upstream AST definitions,
binding ``keras`` to tf_keras for TensorFlow 2.20 compatibility. Removed NumPy
aliases and integer scalar assignments are adapted for NumPy 2. Graph widths and QC/input sizes are
reduced; levels, kernels, skip connections and all prediction operations remain.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import os
import platform
from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Unpack, cast

import numpy as np
from reference_types import (
    ArchiveWriter,
    DefinitionNamespace,
    FloatArray,
    Geometry,
    GraphLabels,
    HasLayers,
    HostArray,
    IndexArray,
    KerasBindings,
    NetworkOptions,
    RealArray,
    ReferenceModel,
    ReferenceNamespace,
    SelectedLabels,
    TensorBindings,
)
from source_admission import require_git_sources

REVISION = "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d"
SOURCE_FILES = (
    "SynthSeg/predict_synthseg.py",
    "SynthSeg/predict.py",
    "ext/neuron/models.py",
    "ext/neuron/utils.py",
    "ext/lab2im/utils.py",
    "ext/lab2im/layers.py",
    "ext/lab2im/edit_tensors.py",
    "ext/lab2im/edit_volumes.py",
)
SEED = 719
FEATURES = 2
DENOISER_FEATURES = 1
SIZE = 16


def provenance() -> dict[str, HostArray]:
    """Validate the recording host and report source provenance."""
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Record references on macOS arm64")
    return dict(
        reference_host=np.array("darwin-arm64"),
        synthseg_commit=np.array(REVISION),
        numpy=np.array(np.__version__),
        scipy=np.array(version("scipy")),
        nibabel=np.array(version("nibabel")),
        source_paths=np.array(list(SOURCE_FILES)),
    )


class NumpyAliases(ast.NodeTransformer):
    """Rewrite deprecated NumPy scalar aliases in loaded source ASTs."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Adapt deprecated NumPy alias references in the loaded source AST."""
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr
            in (
                "int",
                "float",
            )
        ):
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def definitions(
    path: Path, namespace: dict[str, object], names: set[str] | None = None
) -> SimpleNamespace:
    """Extract and compile the required source definitions."""
    code = path.read_text()
    if path.name == "predict.py":
        for rhs in (
            "np.where(labels_segmentation == lab)[0]",
            "lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]",
            "lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]",
        ):
            assert code.count(rhs) == 1
            code = code.replace(rhs, rhs + ".item()")
    if path.name == "edit_volumes.py":
        rhs = "ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]"
        assert code.count(rhs) == 1
        code = code.replace(rhs, rhs + ".item()")
    tree = ast.parse(code)
    tree = NumpyAliases().visit(tree)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (names is None or node.name in names)
    ]
    scope = dict(namespace)
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope)


def load_reference(root: Path, source_repository: Path | None = None) -> ReferenceNamespace:
    """Load source SynthSeg functions and reduced reference networks."""
    import nibabel as nib
    import scipy.ndimage as ndi

    tf = cast(TensorBindings, import_module("tensorflow"))
    keras = cast(KerasBindings, import_module("tf_keras"))
    from scipy.interpolate import RegularGridInterpolator

    labels = [
        f"data/labels_classes_priors/synthseg_{field}{suffix}.npy"
        for suffix in ("", "_2.0")
        for field in (
            "segmentation_labels",
            "segmentation_names",
            "topological_classes",
            "qc_labels",
            "qc_names",
        )
    ] + [
        f"data/labels_classes_priors/synthseg_{field}.npy"
        for field in ("parcellation_labels", "parcellation_names", "denoiser_labels_2.0")
    ]
    require_git_sources(
        root, REVISION, {name: name for name in (*SOURCE_FILES, *labels)}, source_repository
    )
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    ns: dict[str, object] = dict(
        np=np,
        tf=tf,
        keras=keras,
        KL=keras.layers,
        K=keras.backend,
        Model=keras.models.Model,
        Layer=keras.layers.Layer,
        os=os,
        csv=csv,
        math=math,
        combinations=combinations,
        nib=nib,
        scipy_label=ndi.label,
        RegularGridInterpolator=RegularGridInterpolator,
        gaussian_filter=ndi.gaussian_filter,
    )
    utils = definitions(root / "ext/lab2im/utils.py", ns)
    ns["utils"] = utils
    nrn_utils = definitions(root / "ext/neuron/utils.py", ns)
    ns.update(
        nrn_utils=nrn_utils,
        volshape_to_meshgrid=cast(DefinitionNamespace, nrn_utils).volshape_to_meshgrid,
    )
    ns["l2i_et"] = definitions(root / "ext/lab2im/edit_tensors.py", ns, {"gaussian_kernel"})
    ns["layers"] = definitions(
        root / "ext/lab2im/layers.py",
        ns,
        {"GaussianBlur", "ConvertLabels", "RandomFlip"},
    )
    ns["edit_volumes"] = definitions(root / "ext/lab2im/edit_volumes.py", ns)
    labels = definitions(root / "SynthSeg/predict.py", ns, {"get_flip_indices", "write_csv"})
    ns.update(
        get_flip_indices=cast(DefinitionNamespace, labels).get_flip_indices,
        write_csv=cast(DefinitionNamespace, labels).write_csv,
    )
    networks = definitions(root / "ext/neuron/models.py", ns, {"unet", "conv_enc", "conv_dec"})
    original_unet, original_encoder = (
        cast(DefinitionNamespace, networks).unet,
        cast(DefinitionNamespace, networks).conv_enc,
    )

    def reduced_unet(**kwargs: Unpack[NetworkOptions]) -> ReferenceModel:
        """Build a reduced source U-Net for a deterministic fixture."""
        kwargs["nb_features"] = DENOISER_FEATURES if kwargs["name"] == "l2l" else FEATURES
        return original_unet(**kwargs)

    def reduced_encoder(**kwargs: Unpack[NetworkOptions]) -> ReferenceModel:
        """Build a reduced source encoder for a deterministic fixture."""
        kwargs["nb_features"] = FEATURES
        return original_encoder(**kwargs)

    ns["nrn_models"] = SimpleNamespace(unet=reduced_unet, conv_enc=reduced_encoder)
    return cast(ReferenceNamespace, definitions(root / "SynthSeg/predict_synthseg.py", ns))


def seed_checkpoint(
    model: ReferenceModel, path: str, *, cortex_index: int, **_kwargs: object
) -> ReferenceModel:
    """Replace author-weight loading with deterministic float32 synthetic parameters."""

    prefix = Path(path).stem
    for layer in model.layers:
        prefixes = ("unet", "l2l") if prefix == "unet" else (prefix,)
        if not layer.name.startswith(prefixes) or not layer.weights:
            continue
        values: list[FloatArray] = []
        for tensor in layer.weights:
            key = tensor.name.rsplit("/", 1)[-1]
            value, *_ = weight_spec(layer.name, key, tuple(tensor.shape), cortex_index)
            values.append(value)
        layer.set_weights(values)
    return model


def weight_spec(
    name: str,
    key: str,
    shape: tuple[int, ...],
    cortex_index: int,
) -> tuple[FloatArray, int, float, float, int, float]:
    """Derive reproducible synthetic tensor values for one named model weight."""
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{name}:{key}".encode()).digest()[:4], "little")
    scale, offset = 0.1, 0.0
    if key == "kernel:0":
        scale = math.sqrt(6.0 / np.prod(shape[:-1]))
    elif key in ("gamma:0", "moving_variance:0"):
        scale, offset = 0.15, 1.0
    values = np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)
    values = values * np.float32(scale) + np.float32(offset)
    boost_index, boost = -1, 0.0
    if name in ("unet_likelihood", "unet2_likelihood") and key == "bias:0":
        # Keep random reduced-width networks above upstream's 0.2 cleanup threshold
        # and exercise cortical parcellation in the complete runner.
        boost_index, boost = cortex_index, 3.0
        values[boost_index] += np.float32(boost)
    return values, seed, scale, offset, boost_index, boost


def checkpoint_specs(model: HasLayers, cortex_index: int) -> dict[str, HostArray]:
    """Describe seeded tensor layouts and values for each reference checkpoint."""
    records: list[tuple[str, str, list[int], int, float, float, int, float]] = []
    for layer in model.layers:
        for tensor in layer.weights:
            key = tensor.name.rsplit("/", 1)[-1]
            shape = tuple(tensor.shape)
            expected, seed, scale, offset, boost_index, boost = weight_spec(
                layer.name,
                key,
                shape,
                cortex_index,
            )
            if not np.array_equal(tensor.numpy(), expected):
                raise ValueError(f"Reference parameter was not seeded: {layer.name}/{key}")
            records.append(
                (
                    layer.name,
                    key,
                    [*shape, *([-1] * (5 - len(shape)))],
                    seed,
                    scale,
                    offset,
                    boost_index,
                    boost,
                )
            )
    # Nested models occur only in the flipped branch and share the seeded weights.
    records = [
        r
        for r in records
        if ":" not in r[0]
        and r[1]
        in (
            "kernel:0",
            "bias:0",
            "gamma:0",
            "beta:0",
            "moving_mean:0",
            "moving_variance:0",
        )
    ]
    return dict(
        weight_layers=np.array([r[0] for r in records]),
        weight_keys=np.array([r[1] for r in records]),
        weight_shapes=np.array([r[2] for r in records]),
        weight_seeds=np.array([r[3] for r in records], dtype=np.uint32),
        weight_scales=np.array([r[4] for r in records], dtype=np.float32),
        weight_offsets=np.array([r[5] for r in records], dtype=np.float32),
        weight_boost_indices=np.array([r[6] for r in records], dtype=np.int32),
        weight_boost_values=np.array([r[7] for r in records], dtype=np.float32),
    )


def selected_labels(
    root: Path, source: ReferenceNamespace, *, fast: bool, v1: bool
) -> SelectedLabels:
    """Load the source label assets required by the selected variant."""
    data = root / "data/labels_classes_priors"
    suffix = "" if v1 else "_2.0"
    raw = cast(IndexArray, np.load(data / f"synthseg_segmentation_labels{suffix}.npy"))
    if fast:
        segmentation, indices = np.unique(raw, return_index=True)
        flips = None
    else:
        segmentation, flips, indices = source.get_flip_indices(raw, 18 if v1 else 19)
    return cast(
        SelectedLabels,
        dict(
            labels_segmentation=segmentation,
            flip_indices=flips,
            labels_denoiser=np.unique(np.load(data / "synthseg_denoiser_labels_2.0.npy")),
            labels_parcellation=np.array([0, 1001, 1002, 2001, 2002], dtype=np.int32),
            labels_qc=np.load(data / f"synthseg_qc_labels{suffix}.npy")[indices],
            topology_classes=np.load(data / f"synthseg_topological_classes{suffix}.npy")[indices],
        ),
    )


def build_reference_model(
    source: ReferenceNamespace, labels: SelectedLabels, robust: bool
) -> ReferenceModel:
    """Construct a selected reduced source graph and seed its checkpoint."""
    graph_labels = cast(
        GraphLabels,
        {
            key: value
            for key, value in cast(Mapping[str, IndexArray | None], labels).items()
            if key != "topology_classes"
        },
    )
    original_load = source.Model.load_weights
    cortex_index = np.flatnonzero(labels["labels_segmentation"] == 3).item()

    def load_seeded(model: ReferenceModel, path: str, **kwargs: object) -> ReferenceModel:
        """Populate a source model with deterministic checkpoint tensors."""
        return seed_checkpoint(model, path, cortex_index=cortex_index, **kwargs)

    source.Model.load_weights = load_seeded
    try:
        with TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("unet", "unet_parc", "qc")]
            for path in paths:
                path.touch()
            model = source.build_model(
                *(str(path) for path in paths),
                input_shape_qc=SIZE,
                sigma_smoothing=0.5,
                robust=robust,
                do_parcellation=True,
                do_qc=True,
                **graph_labels,
            )
    finally:
        source.Model.load_weights = original_load
    return model


def postprocess_reference(
    source: ReferenceNamespace,
    predictions: list[FloatArray],
    labels: SelectedLabels,
    fast: bool,
    v1: bool,
    **geometry: Unpack[Geometry],
) -> tuple[IndexArray, RealArray, RealArray]:
    """Run source postprocessing for the recorded variant."""
    return source.postprocess(
        predictions[0].copy(),
        predictions[1].copy(),
        labels_segmentation=labels["labels_segmentation"],
        labels_parcellation=labels["labels_parcellation"],
        fast=fast,
        topology_classes=labels["topology_classes"],
        v1=v1,
        **geometry,
    )


def record_runner(
    root: Path,
    source: ReferenceNamespace,
    model: ReferenceModel,
    labels: SelectedLabels,
    fast: bool,
    v1: bool,
    *,
    ct: bool = False,
    crop: list[int] | None = None,
    raw_shape: int = SIZE,
) -> dict[str, HostArray]:
    """Record the selected source runner outputs and geometry."""
    data = root / "data/labels_classes_priors"
    suffix = "" if v1 else "_2.0"
    _, indices = np.unique(
        np.load(data / f"synthseg_segmentation_labels{suffix}.npy"), return_index=True
    )
    names = np.load(data / f"synthseg_segmentation_names{suffix}.npy")[indices]
    raw_parc = np.load(data / "synthseg_parcellation_labels.npy")
    parc_names = np.load(data / "synthseg_parcellation_names.npy")
    parc_names = parc_names[
        [np.flatnonzero(raw_parc == value)[0] for value in labels["labels_parcellation"][1:]]
    ]
    volume_labels = np.concatenate(
        (labels["labels_segmentation"], labels["labels_parcellation"][1:])
    )
    volume_names = np.concatenate((names, parc_names))
    if not v1:
        volume_labels = np.append(volume_labels, np.max(volume_labels) + 1)
        volume_names = np.append(volume_names, "total intracranial")
    raw = np.random.default_rng(SEED + 1).uniform(-100, 250, (raw_shape,) * 3).astype(np.float32)
    affine = np.eye(4)
    if ct:
        affine = np.array([[0, -0.8, 0, 8], [0, 0, 1.3, -9], [-1.2, 0, 0, 10], [0, 0, 0, 1.0]])
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "image.nii.gz"
        resampled_path = Path(temporary) / "resampled.nii.gz"
        source.nib.save(source.nib.Nifti1Image(raw, affine), path)
        input_affine = source.nib.load(path).affine
        # Upstream's CLI sets min_pad to the crop size when --crop is given, and to
        # 128 (here the reduced SIZE) otherwise (SynthSeg_predict.py / predict_synthseg.py).
        image, aff, _, resolution, shape, pad, crop_idx = source.preprocess(
            str(path),
            ct=ct,
            n_levels=4,
            min_pad=crop if crop is not None else raw_shape,
            crop=crop,
            path_resample=str(resampled_path),
        )
        resampled = raw
        if ct:
            resampled = cast(FloatArray, np.asanyarray(source.nib.load(resampled_path).dataobj))
    predictions = [
        value.numpy()
        for value in model(
            [image.astype(np.float32), np.array([image.shape[1:-1]], dtype=np.int32)],
            training=False,
        )
    ]
    mask, _, volumes = postprocess_reference(
        source,
        predictions,
        labels,
        fast,
        v1,
        shape=shape,
        pad_idx=pad,
        crop_idx=crop_idx,
        aff=aff,
        im_res=resolution,
    )
    return dict(
        runner_input=raw,
        runner_mask=mask,
        runner_volumes=volumes,
        runner_qc=predictions[2],
        runner_input_affine=input_affine,
        runner_affine=aff,
        runner_resampled=resampled,
        volume_labels=volume_labels,
        volume_names=volume_names,
    )


def record_variant(root: Path, output: Path, source: ReferenceNamespace, name: str) -> None:
    """Record graph and processing outputs for one selected variant."""
    robust, fast, v1 = name == "robust", name in ("fast", "robust", "v1_fast"), "v1" in name
    labels = selected_labels(root, source, fast=fast, v1=v1)
    model = build_reference_model(source, labels, robust)
    image = np.random.default_rng(SEED).uniform(0, 1, (1, SIZE, SIZE, SIZE, 1)).astype(np.float32)
    shape = np.array([[SIZE] * 3], dtype=np.int32)
    predictions = [value.numpy() for value in model([image, shape], training=False)]
    segmentation, parcellation, qc = predictions
    geometry = cast(
        Geometry,
        dict(
            shape=[SIZE] * 3,
            pad_idx=np.array([0, 0, 0, SIZE, SIZE, SIZE]),
            crop_idx=None,
            aff=np.eye(4),
            im_res=np.ones(3),
        ),
    )
    mask, posteriors, volumes = postprocess_reference(
        source,
        predictions,
        labels,
        fast,
        v1,
        **geometry,
    )
    parc_input = source.Model(model.inputs, model.get_layer("unet_parc_conv_downarm_0_0").input)
    parcellation_input = parc_input([image, shape], training=False).numpy()
    source_layers = [*model.layers]
    # A nested flipped segmentation model shares the outer graph's parameters.
    specs = checkpoint_specs(
        cast(
            HasLayers,
            SimpleNamespace(
                layers=[
                    layer
                    for layer in source_layers
                    if isinstance(layer, (source.KL.Conv3D, source.KL.BatchNormalization))
                ]
            ),
        ),
        np.flatnonzero(labels["labels_segmentation"] == 3).item(),
    )
    arrays: dict[str, HostArray] = dict(
        **provenance(),
        tensorflow=np.array(source.tf.__version__),
        tf_keras=np.array(source.keras.__version__),
        reference_device=np.array("CPU"),
        seed=np.array(SEED),
        features=np.array(FEATURES),
        denoiser_features=np.array(DENOISER_FEATURES),
        qc_shape=np.array(SIZE),
        robust=np.array(robust),
        fast=np.array(fast),
        v1=np.array(v1),
        input=image,
        segmentation=segmentation,
        parcellation=parcellation,
        qc=qc,
        mask=mask,
        posteriors=posteriors,
        volumes=volumes,
        parcellation_input=parcellation_input,
        **record_runner(root, source, model, labels, fast, v1),
        **specs,
        **{
            key: value if value is not None else np.array([], dtype=np.int32)
            for key, value in cast(Mapping[str, IndexArray | None], labels).items()
        },
    )
    if name == "fast":
        arrays.update(
            {
                f"ct_{key}": value
                for key, value in record_runner(
                    root, source, model, labels, fast, v1, ct=True
                ).items()
                if key.startswith("runner_")
            }
        )
        # Exercises the --crop CLI option: a raw image bigger than the requested
        # patch, so crop_idx is a genuine sub-region rather than the full volume.
        arrays.update(
            {
                f"crop_{key}": value
                for key, value in record_runner(
                    root, source, model, labels, fast, v1, crop=[16, 16, 16], raw_shape=20
                ).items()
                if key.startswith("runner_")
            }
        )
    path = output / f"{name}.npz"
    cast(ArchiveWriter, np.savez_compressed)(path, **arrays)
    if path.stat().st_size >= 1_000_000:
        raise ValueError(f"Fixture exceeds 1 MB: {path}")
    print(f"{name}: {path.stat().st_size} bytes")


def main() -> None:
    """Record pinned upstream fixtures for the requested SynthSeg variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthseg-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/variants"))
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    provenance()
    source = load_reference(args.synthseg_src, args.source_repository)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("standard", "fast", "robust", "v1", "v1_fast"):
        record_variant(args.synthseg_src, args.output, source, name)


if __name__ == "__main__":
    main()
