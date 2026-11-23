"""SynthSeg postprocessing on the Metal GPU, mirroring ``processing.postprocess``.

The network posteriors stay in MLX; the int32 label map, the optional posteriors
and the per-label volume sums are copied to the host slab by slab, where pasting
and reorientation use the reference NumPy helpers. Labels and
posteriors follow the NumPy reference; posterior normalization sums channels in
MLX's order, so posteriors can differ from NumPy by float32 rounding. Volume sums
are exact to float64 level and reported in float64 for every variant (see
``_block_sums``); upstream accumulates and reports uncropped non-Standard runs
in float32, which on a 1 mm brain volume moves volumes by up to tens of 0.001 mm^3.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import mlx.core as mx
import numpy as np
from medmlx_core.typing import MetalKernel

from synthseg_mlx import processing
from synthseg_mlx._lab2im import edit_volumes
from synthseg_mlx._typing import (
    BoolArray,
    DoubleArray,
    FloatArray,
    IndexArray,
    LabelArray,
    NativeArrayApi,
    RealArray,
)
from synthseg_mlx.processing import report_volumes

# Blocks of 4096 voxels keep the split partial sums exact in float32 (see below).
_BLOCK = 4096
# About 1M voxels per slab: 33 float32 channels make one 135 MB slab copy.
_SLAB_VOXELS = 2**20


@lru_cache(maxsize=1)
def _split_sum_kernel() -> MetalKernel:
    # Each posterior p in [0, 1] splits exactly into hi (multiple of 2**-12),
    # mid (multiple of 2**-24, |mid| <= 2**-13) and lo (|lo| <= 2**-25). Over 4096
    # voxels the hi and mid partial sums stay on their grids within 24 bits, so
    # they are exact in any order (fast math cannot change them); lo carries the
    # only rounding, below 1e-7 voxel per block. The host adds the block sums in
    # float64.
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="synthseg_split_sums",
            input_names=["x"],
            output_names=["sums"],
            source="""
        uint c = thread_position_in_grid.x, block = thread_position_in_grid.y;
        if (c >= C || block >= BLOCKS) return;
        float hi = 0.0f, mid = 0.0f, lo = 0.0f;
        ulong stop = metal::min(ulong(block + 1) * B, ulong(V));
        for (ulong v = ulong(block) * B; v < stop; ++v) {
            float value = x[v * C + c];
            float h = metal::rint(value * 4096.0f) * 0.000244140625f;
            float r = value - h;
            float m = metal::rint(r * 16777216.0f) * 5.9604644775390625e-8f;
            hi += h; mid += m; lo += r - m;
        }
        ulong base = ulong(block) * 3 * C + c;
        sums[base] = hi; sums[base + C] = mid; sums[base + 2 * C] = lo;
        """,
        ),
    )


def _block_sums(posteriors: mx.array) -> mx.array:
    """Split block sums ``(3 * blocks, C)`` of ``(..., C)`` posteriors in [0, 1].

    Adding all rows in float64 on the host gives the per-channel totals; block
    sums from separate calls can be added the same way.
    """
    channels = posteriors.shape[-1]
    voxels = posteriors.size // channels
    blocks = max(1, -(-voxels // _BLOCK))
    (sums,) = _split_sum_kernel()(
        inputs=[mx.reshape(posteriors, (voxels, channels))],
        output_shapes=[(blocks * 3, channels)],
        output_dtypes=[mx.float32],
        grid=(channels, blocks, 1),
        threadgroup=(min(channels, 32), 8, 1),
        template=[("C", channels), ("V", voxels), ("B", _BLOCK), ("BLOCKS", blocks)],
        stream=mx.gpu,
    )
    return sums


def largest_component(mask: mx.array) -> mx.array:
    """Largest connected component of a boolean mask (6-connectivity), all False if empty.

    The 3D mask (one byte per voxel) is labelled on the host with the reference
    SciPy routine, which keeps component numbering and tie-breaking identical.
    """
    host = cast(BoolArray, np.array(mask))
    return mx.array(processing.largest_component(host).view(np.uint8)).astype(mx.bool_)


def _normalize(posteriors: mx.array) -> mx.array:
    # 0/0 gives NaN rows, as in NumPy; argmax then returns channel 0 in both.
    return mx.divide(posteriors, mx.sum(posteriors, axis=-1, keepdims=True))


def _slabs(depth: int, rows: int) -> list[tuple[int, int]]:
    return [(z, min(z + rows, depth)) for z in range(0, depth, rows)]


def postprocess_mlx(
    post_patch_seg: mx.array,
    post_patch_parc: mx.array | None,
    shape: Sequence[int],
    pad_idx: IndexArray,
    crop_idx: IndexArray | None,
    labels_segmentation: IndexArray,
    labels_parcellation: IndexArray,
    aff: DoubleArray,
    im_res: RealArray,
    fast: bool,
    topology_classes: IndexArray | None,
    v1: bool,
    return_posteriors: bool = True,
) -> tuple[LabelArray, RealArray | None, RealArray]:
    """GPU ``processing.postprocess``: same arguments (MLX posteriors) and outputs.

    Returns host arrays: the int32 label map, float32 posteriors (when
    ``return_posteriors``) and the rounded mm^3 volume row, computed in float64 for
    every variant, so there is no ``volume_dtype`` option.

    Only the 3D brain and topological-class masks span the whole grid, so
    connectivity matches the reference. Per-channel work runs in slabs of about
    ``_SLAB_VOXELS`` voxels along the first axis, which bounds the device memory
    added to the network outputs to a few slab-sized copies.
    """
    seg = mx.squeeze(post_patch_seg)
    parc = None if post_patch_parc is None else mx.squeeze(post_patch_parc)
    cleanup = not fast and topology_classes is not None
    low = [int(v) for v in pad_idx[:3]]
    size = [int(pad_idx[a + 3]) - low[a] for a in range(3)]
    rows = max(1, _SLAB_VOXELS // (seg.shape[1] * seg.shape[2]))
    eval_ = cast(NativeArrayApi, mx).eval

    # Masks live on the padded grid with topology cleanup, as upstream labels
    # components before cropping, and on the cropped grid otherwise.
    grid = (
        seg
        if cleanup
        else seg[low[0] : low[0] + size[0], low[1] : low[1] + size[1], low[2] : low[2] + size[2]]
    )
    origin = low if cleanup else [0, 0, 0]

    def foreground(z0: int, z1: int) -> mx.array:
        return mx.sum(grid[z0:z1, ..., 1:].astype(mx.float32), axis=-1) > 0.25

    # keep biggest connected component
    brain = largest_component(mx.concatenate([foreground(*s) for s in _slabs(grid.shape[0], rows)]))

    columns = [mx.ones(grid.shape[:3], dtype=mx.bool_)]
    class_index = np.zeros(seg.shape[-1], dtype=np.int32)
    if cleanup:
        classes_of = cast(IndexArray, topology_classes)
        classes = np.unique(classes_of)
        class_index = np.searchsorted(classes, classes_of).astype(np.int32)
        members = [mx.array(np.where(classes_of == k)[0]) for k in classes[1:]]
        # Each topological class's mask reads only its own channels, so the classes
        # are independent; channels of the first class are kept everywhere.
        parts: list[list[mx.array]] = [[] for _ in members]
        for z0, z1 in _slabs(grid.shape[0], rows):
            kept = mx.concatenate(
                [
                    grid[z0:z1, ..., :1] > 0.25,
                    mx.logical_and(grid[z0:z1, ..., 1:] > 0.25, brain[z0:z1, ..., None]),
                ],
                axis=-1,
            )
            for part, channels in zip(parts, members, strict=True):
                part.append(mx.any(mx.take(kept, channels, axis=-1), axis=-1))
            eval_(*(part[-1] for part in parts))
        columns += [largest_component(mx.concatenate(part)) for part in parts]
    masks = mx.stack(columns, axis=-1)
    channel_class = mx.array(class_index)
    seg_labels = mx.array(labels_segmentation.astype(np.int32))
    parc_labels = mx.array(labels_parcellation.astype(np.int32))

    labels = np.empty(size, dtype=np.int32)
    cropped = np.empty([*size, seg.shape[-1]], dtype=np.float32) if return_posteriors else None
    seg_sums: list[mx.array] = []
    parc_sums: list[mx.array] = []
    for z0, z1 in _slabs(size[0], rows):
        window = (
            slice(low[0] + z0, low[0] + z1),
            slice(low[1], low[1] + size[1]),
            slice(low[2], low[2] + size[2]),
        )
        mask_window = (
            slice(origin[0] + z0, origin[0] + z1),
            slice(origin[1], origin[1] + size[1]),
            slice(origin[2], origin[2] + size[2]),
        )
        post = seg[window].astype(mx.float32)
        region = brain[mask_window][..., None]
        if cleanup:
            keep = mx.take(masks[mask_window], channel_class, axis=-1)
        else:
            keep = mx.concatenate(
                [mx.ones((*post.shape[:3], 1), dtype=mx.bool_), post[..., 1:] > 0.2], axis=-1
            )
        # Labels without topology cleanup still obey the brain mask.
        keep = mx.concatenate([keep[..., :1], mx.logical_and(keep[..., 1:], region)], axis=-1)
        post = _normalize(mx.multiply(post, keep))
        slab = mx.take(seg_labels, mx.argmax(post, axis=-1))
        seg_sums.append(_block_sums(post[..., 1:]))
        if parc is not None:
            cortex = mx.logical_or(mx.equal(slab, 3), mx.equal(slab, 42))
            parcels = parc[window].astype(mx.float32)
            parcels = _normalize(
                mx.concatenate(
                    [mx.logical_not(cortex)[..., None].astype(mx.float32), parcels[..., 1:]],
                    axis=-1,
                )
            )
            slab = mx.where(cortex, mx.take(parc_labels, mx.argmax(parcels, axis=-1)), slab)
            parc_sums.append(_block_sums(parcels[..., 1:]))
        eval_(slab, seg_sums[-1], *parc_sums[-1:], *([post] if cropped is not None else []))
        labels[z0:z1] = np.array(slab)
        if cropped is not None:
            cropped[z0:z1] = np.array(post)

    # paste patches back to matrix of original image size, then align prediction
    # back to first orientation, as the reference does on the host
    posteriors = cropped
    if crop_idx is not None:
        crop = tuple(slice(int(crop_idx[a]), int(crop_idx[a + 3])) for a in range(3))
        seg_map = np.zeros(shape=shape, dtype=np.int32)
        seg_map[crop] = labels
        labels = seg_map
        if cropped is not None:
            posteriors = np.zeros(shape=[*shape, cropped.shape[-1]], dtype=np.float32)
            posteriors[..., 0] = 1
            posteriors[crop] = cropped
    label_map = edit_volumes.align_volume_to_ref(
        labels, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False
    )
    if posteriors is not None:
        posteriors = edit_volumes.align_volume_to_ref(
            posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False
        )

    # Background-only margins contribute no foreground volume, so the cropped,
    # unaligned posteriors give the same sums.
    def total(sums: list[mx.array]) -> RealArray:
        blocks = [cast(FloatArray, np.array(block)) for block in sums]
        return np.concatenate(blocks).astype(np.float64).sum(axis=0)

    volumes_parc = total(parc_sums) if parc_sums else None
    return (
        label_map,
        cast(RealArray | None, posteriors),
        report_volumes(total(seg_sums), volumes_parc, labels_segmentation, im_res, v1),
    )
