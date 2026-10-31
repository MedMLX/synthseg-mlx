"""Array-only inference for the SynthSeg MLX executor.

Callers own image I/O, label selection, and output writing. Preprocessing and
postprocessing use NumPy/SciPy code extracted from SynthSeg (see THIRD_PARTY_NOTICES.md).
Volumes are channels-last (NDHWC) and, inside the network, in RAS axis order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from medmlx_core.runtime import import_mlx

from synthseg_mlx._typing import (
    DoubleArray,
    FloatArray,
    IndexArray,
    LabelArray,
    NativeArrayApi,
    RealArray,
)
from synthseg_mlx.processing import postprocess
from synthseg_mlx.variants import SynthSegVariant, parse_variant

if TYPE_CHECKING:
    from mlx.core import array as NativeArray

    from synthseg_mlx.mlx_model import SynthSegModel


def prepare_network_input(image: RealArray) -> NativeArray:
    """Copy an already preprocessed NDHWC image ``(1, D, H, W, 1)`` to MLX as contiguous float32."""

    import_mlx()
    import mlx.core as mlxc

    return mlxc.array(np.ascontiguousarray(image, dtype=np.float32))


def predict_numpy(
    model: SynthSegModel, image: RealArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Run all three graphs and return segmentation/parcellation posteriors and QC.

    ``image`` is the source-preprocessed, padded NDHWC single-channel volume
    ``(1, D, H, W, 1)`` from ``preprocess``: intensities rescaled to [0, 1], RAS axis
    order, and D, H, W divisible by 16 (``preprocess`` pads to multiples of 32).
    The networks are evaluated stage by stage (``SynthSegModel.predict``) to
    bound peak memory; the final ``eval`` keeps the explicit host-copy boundary.

    Returns:
        Float32 host arrays ``(segmentation, parcellation, qc)``. ``segmentation`` is
        ``(1, D, H, W, n_segmentation_labels)`` posteriors in the order of the sorted
        label list given to the model, background first; ``parcellation`` is
        ``(1, D, H, W, n_parcellation_labels)``; ``qc`` is ``(1, n_qc_classes)`` raw
        regional scores for ``qc_scores``. Spatial axes are those of the padded input.

    Raises:
        ModelExecutionError: If ``image`` is not a single-channel NDHWC volume, or its
            grid cannot be reproduced by the decoder.
    """

    import_mlx()
    import mlx.core as mlxc

    network_input = prepare_network_input(image)
    segmentation, parcellation, qc_score = model.predict(network_input)
    cast(NativeArrayApi, mlxc).eval(segmentation, parcellation, qc_score)
    return (
        cast(FloatArray, np.array(segmentation)),
        cast(FloatArray, np.array(parcellation)),
        cast(FloatArray, np.array(qc_score)),
    )


def postprocess_arrays(
    segmentation: RealArray,
    parcellation: RealArray | None,
    *,
    shape: list[int],
    pad_idx: IndexArray,
    crop_idx: IndexArray | None,
    labels_segmentation: IndexArray,
    labels_parcellation: IndexArray,
    affine: DoubleArray,
    image_resolution: RealArray,
    topology_classes: IndexArray | None,
    robust: bool = False,
    fast: bool = False,
    v1: bool = False,
) -> tuple[LabelArray, RealArray | None, RealArray]:
    """Turn network posteriors into a label map, posteriors and volumes, as upstream does.

    Works on copies of ``segmentation`` and ``parcellation``, so caller storage is not
    modified.

    Args:
        segmentation: Posteriors ``(1, D, H, W, n_segmentation_labels)`` on the padded grid.
        parcellation: Cortical parcellation posteriors on the same grid, or ``None`` to
            leave the cortex labels (3 and 42) unparcellated.
        shape: Spatial shape after RAS alignment, before cropping and padding
            (``preprocess`` output).
        pad_idx: Padding indices from ``preprocess``: lower bounds then upper bounds of
            the original volume inside the padded one.
        crop_idx: Cropping indices from ``preprocess``, or ``None`` if not cropped.
        labels_segmentation: Sorted unique segmentation label values, one per channel.
        labels_parcellation: Sorted unique parcellation label values, background first.
        affine: Affine from ``preprocess``; the outputs are re-oriented to match it.
        image_resolution: Voxel size in mm ``(3,)`` from ``preprocess``; scales volumes
            to mm^3.
        topology_classes: Topological class per segmentation label for the per-class
            largest-component cleanup, or ``None`` to skip it.
        robust: Select the robust variant.
        fast: Select the fast variant (skips the per-class cleanup).
        v1: Select SynthSeg 1.0 (no total intracranial volume).

    Returns:
        ``(label_map, posteriors, volumes)``. ``label_map`` is int32 FreeSurfer labels,
        with parcels in place of cortex labels when ``parcellation`` is given, in the
        orientation of ``affine``. ``posteriors`` are the cleaned, renormalized
        segmentation posteriors in the same orientation; they are returned even though
        upstream's ``return_posteriors`` flag is off. ``volumes`` are soft (posterior)
        volumes in mm^3 rounded to three decimals: total intracranial first (not for
        v1), then each non-background segmentation label, then left and right parcels
        when parcellating, scaled to sum to each hemisphere's cortex volume. Standard
        accumulates in float64; the other variants follow upstream's float32. A zero
        parcellation posterior sum for a hemisphere can produce NaN parcel volumes.
    """

    variant = parse_variant(robust=robust, fast=fast, v1=v1)
    return postprocess(
        post_patch_seg=np.array(segmentation),
        post_patch_parc=np.array(parcellation) if parcellation is not None else None,
        shape=shape,
        pad_idx=pad_idx,
        crop_idx=crop_idx,
        labels_segmentation=labels_segmentation,
        labels_parcellation=labels_parcellation,
        aff=affine,
        im_res=image_resolution,
        fast=variant.fast,
        topology_classes=topology_classes,
        v1=variant.v1,
        reuse_segmentation=True,
        return_posteriors=False,
        volume_dtype=np.float64 if variant is SynthSegVariant.STANDARD else None,
    )


def qc_scores(
    qc_score: RealArray,
    *,
    robust: bool = False,
    fast: bool = False,
    v1: bool = False,
) -> RealArray:
    """Clip raw QC output ``(1, n_qc_classes)`` to [0, 1] and round to 4 places.

    The background class is dropped, so the result has ``n_qc_classes - 1`` regional
    scores; at least two classes are expected. Standard casts to float64 before
    clipping and rounding; the other variants retain the input dtype. NaNs propagate.
    """

    variant = parse_variant(robust=robust, fast=fast, v1=v1)
    dtype = np.float64 if variant is SynthSegVariant.STANDARD else None
    raw_quality = np.asarray(np.array(qc_score), dtype=dtype)
    return np.around(np.clip(np.squeeze(raw_quality)[1:], 0, 1), 4)


__all__ = ["postprocess_arrays", "predict_numpy", "prepare_network_input", "qc_scores"]
