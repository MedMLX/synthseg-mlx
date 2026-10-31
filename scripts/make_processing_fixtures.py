"""Record upstream preprocessing, postprocessing, labels and QC crops on macOS arm64.

Uses the upstream loader from make_variant_fixtures, never the MLX port.
Run in the same separate TensorFlow reference environment as that recorder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from make_variant_fixtures import load_reference, provenance
from reference_types import (
    ArchiveValue,
    ArchiveWriter,
    Geometry,
    HostArray,
    RealArray,
    ReferenceNamespace,
)

SEED = 733


def save(output: Path, name: str, **arrays: ArchiveValue) -> None:
    """Write a named source-processing fixture with metadata."""
    path = output / f"{name}.npz"
    cast(ArchiveWriter, np.savez_compressed)(path, **provenance(), **arrays)
    if path.stat().st_size >= 1_000_000:
        raise ValueError(f"Fixture exceeds 1 MB: {path}")
    print(f"{name}: {path.stat().st_size} bytes")


def record_preprocessing(source: ReferenceNamespace, output: Path, name: str) -> None:
    """Record seeded source image preparation cases."""
    rng = np.random.default_rng(SEED)
    raw = rng.uniform(-100, 250, (19, 21, 23)).astype(np.float32)
    affine, target, crop = np.eye(4), np.ones(3), None
    if name == "resample":
        affine = np.diag([1.2, 1.8, 0.7, 1.0])
    elif name == "orientation":
        affine = np.array([[0, -1, 0, 8], [0, 0, 1, -9], [-1, 0, 0, 10], [0, 0, 0, 1.0]])
    elif name == "crop_ct":
        crop = [16, 16, 16]
    elif name == "constant":
        raw.fill(7)
    elif name == "downsample":
        raw = rng.uniform(-100, 250, (25, 27, 33)).astype(np.float32)
        affine, target = np.diag([1, 1.2, 0.8, 1.0]), np.full(3, 2.0)
    elif name == "multichannel":
        raw = np.stack([raw, raw * 2], axis=-1)
    with TemporaryDirectory() as temporary:
        path, resampled = Path(temporary) / "image.nii.gz", Path(temporary) / "resampled.nii.gz"
        source.nib.save(source.nib.Nifti1Image(raw, affine), path)
        loaded, _, affine, _, _, _, resolution = source.utils.get_volume_info(str(path), True)
        image, aff, _, res, shape, pad, indices = source.preprocess(
            str(path),
            ct=name == "crop_ct",
            target_res=target,
            n_levels=3,
            crop=crop,
            min_pad=16,
            path_resample=str(resampled),
        )
        sampled: dict[str, HostArray] = {}
        if resampled.exists():
            volume = source.nib.load(resampled)
            sampled = dict(
                resampled_image=cast(RealArray, np.asanyarray(volume.dataobj)),
                resampled_affine=volume.affine,
            )
    save(
        output,
        f"preprocess_{name}",
        input=loaded,
        affine=affine,
        image_resolution=resolution,
        target_res=target,
        n_levels=3,
        min_pad=16,
        ct=name == "crop_ct",
        crop=np.array([] if crop is None else crop),
        expected_image=image,
        expected_affine=aff,
        expected_resolution=res,
        expected_shape=shape,
        expected_pad_idx=pad,
        expected_crop_idx=np.array([] if indices is None else indices),
        **sampled,
    )


def record_postprocessing(source: ReferenceNamespace, output: Path, name: str) -> None:
    """Record seeded source segmentation and parcellation cases."""
    rng = np.random.default_rng(SEED)
    seg = np.empty((1, 16, 16, 16, 3), np.float32)
    seg[:] = [0.9, 0.05, 0.05]
    region = rng.uniform(0.3, 1, (9, 9, 9, 3)).astype(np.float32)
    region[..., 0] = 0.01
    seg[0, 3:12, 3:12, 3:12] = region / region.sum(axis=-1, keepdims=True)
    seg[0, 3:8, 3:12, 3:12] = [0.01, 0.8, 0.19]
    seg[0, 8:12, 3:12, 3:12] = [0.01, 0.19, 0.8]
    seg[0, 9:11, 5:7, 5:7] = [0.01, 0.8, 0.19]  # A separate class island inside the brain.
    seg[0, 0, 0, 0] = [0.01, 0.8, 0.19]  # Disconnected foreground is removed even in fast mode.
    parc = rng.uniform(0.1, 1, (1, 16, 16, 16, 5)).astype(np.float32)
    parc /= parc.sum(axis=-1, keepdims=True)
    crop = np.array([2, 3, 4, 18, 19, 20]) if name == "crop" else None
    topology = None if name == "no_topology" else np.array([0, 1, 2])
    geometry = cast(
        Geometry,
        dict(
            shape=[20, 22, 24] if crop is not None else [16] * 3,
            pad_idx=np.array([0, 0, 0, 16, 16, 16]),
            crop_idx=crop,
            aff=np.eye(4),
            im_res=np.ones(3),
        ),
    )
    labels_seg, labels_parc = np.array([0, 3, 42]), np.array([0, 1001, 1002, 2001, 2002])
    mask, posteriors, volumes = source.postprocess(
        seg.copy(),
        parc.copy() if name != "v1_no_parc" else None,
        labels_segmentation=labels_seg,
        labels_parcellation=labels_parc,
        fast=name == "fast",
        topology_classes=topology,
        v1=name == "v1_no_parc",
        **geometry,
    )
    save(
        output,
        f"postprocess_{name}",
        segmentation=seg,
        parcellation=parc,
        has_parcellation=name != "v1_no_parc",
        fast=name == "fast",
        v1=name == "v1_no_parc",
        labels_segmentation=labels_seg,
        labels_parcellation=labels_parc,
        topology_classes=np.array([] if topology is None else topology),
        **{key: np.array([] if value is None else value) for key, value in geometry.items()},
        expected_mask=mask,
        expected_posteriors=posteriors,
        expected_volumes=volumes,
    )


def record_labels(root: Path, output: Path) -> None:
    """Record source label ordering and associated names."""
    names = [
        f"synthseg_{field}{suffix}.npy"
        for suffix in ("", "_2.0")
        for field in (
            "segmentation_labels",
            "segmentation_names",
            "topological_classes",
            "qc_labels",
            "qc_names",
        )
    ]
    names += [
        "synthseg_parcellation_labels.npy",
        "synthseg_parcellation_names.npy",
        "synthseg_denoiser_labels_2.0.npy",
    ]
    paths = sorted(root / "data/labels_classes_priors" / name for name in names)
    save(
        output,
        "labels",
        asset_names=np.array([path.name for path in paths]),
        **{path.stem: np.load(path, allow_pickle=False) for path in paths},
    )


def record_qc_crops(source: ReferenceNamespace, output: Path) -> None:
    """Record source quality-control crop cases."""
    values = np.zeros((3, 19, 21, 23), dtype=np.int32)
    values[0, 2:12, 4:15, 6:17] = 3
    values[1, :6, :7, :8] = 3
    values[:, 8, 9, 10] = 24
    for target in (8, 32):
        cropped = source.MakeShape(target)(
            [source.tf.constant(values), source.tf.constant([[19, 21, 23]] * 3)],
        ).numpy()
        save(output, f"qc_crop_{target}", input=values, expected=cropped, target=target)


def main() -> None:
    """Record source preprocessing, postprocessing, label, and QC fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthseg-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/processing"))
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    provenance()
    source = load_reference(args.synthseg_src, args.source_repository)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("resample", "orientation", "crop_ct", "constant", "downsample", "multichannel"):
        record_preprocessing(source, args.output, name)
    for name in ("standard", "crop", "fast", "no_topology", "v1_no_parc"):
        record_postprocessing(source, args.output, name)
    record_labels(args.synthseg_src, args.output)
    record_qc_crops(source, args.output)


if __name__ == "__main__":
    main()
