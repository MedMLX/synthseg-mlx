"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

from __future__ import annotations

# Extracted subset; imports and package boundaries adapted for synthseg-mlx.
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import find_objects

from synthseg_mlx._lab2im import edit_volumes, utils
from synthseg_mlx._typing import (
    BoolArray,
    DoubleArray,
    IndexArray,
    LabelArray,
    ParcellationPostprocessor,
    RealArray,
    Resolution,
    SegmentationPostprocessor,
    Size,
)


def preprocess(
    image: RealArray,
    affine: DoubleArray,
    image_resolution: RealArray,
    ct: bool,
    target_res: Resolution = 1.,
    n_levels: int = 5,
    crop: Size | None = None,
    min_pad: Size | None = None,
    *,
    on_resample: Callable[[RealArray, DoubleArray], None] | None = None,
) -> tuple[RealArray, DoubleArray, RealArray, list[int], IndexArray, IndexArray | None]:
    """Preprocess a loaded, squeezed volume without image I/O.

    Pass the loader's original dtype, affine and voxel sizes without recasting.
    Return (image, affine, resolution, shape, pad_idx, crop_idx). The optional
    on_resample(image, affine) hook lets the caller save the resampled image
    before orientation, cropping and normalization, at the source I/O boundary.

    Steps, in order: resample to ``target_res`` mm when any voxel size differs by more
    than 0.05 mm; reorient the axes to RAS (identity reference orientation); optionally center-crop to
    ``crop`` (each size rounded up to a multiple of ``2 ** n_levels``); clip CT to
    [0, 80] if ``ct``; clip to the 0.5th and 99.5th percentiles and rescale to [0, 1];
    zero-pad, centered, to a multiple of ``2 ** n_levels`` (32) of at least ``min_pad``
    (supply it, as the medmlx runner does); add batch and channel axes.

    Returns:
        ``image`` is ``(1, D, H, W, 1)`` in RAS axis order. ``affine`` and ``resolution``
        describe the (possibly resampled) input before reorientation, which
        ``postprocess`` needs to restore its orientation. ``shape`` is the spatial shape
        after reorientation, before cropping and padding. ``pad_idx`` and ``crop_idx``
        hold lower bounds then upper bounds per axis (``crop_idx`` is ``None`` without
        ``crop``).

    Raises:
        Exception: A bare ``Exception``, as in the source, if the input is not a 3D
            volume (or a 2D one with too few slices).
    """

    # Image I/O stays with the caller; retain the source dimension inference.
    im, aff, im_res = image, affine, image_resolution
    n_dims, n_channels = utils.get_dims(list(im.shape))
    if n_dims == 2 and 1 < n_channels < 4:
        raise Exception('either the input is 2D with several channels, or is 3D with at most 3 slices. '
                        'Either way, results are going to be poor...')
    elif n_dims == 2 and 3 < n_channels < 11:
        print('warning: input with very few slices')
        n_dims = 3
    elif n_dims < 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # resample image if necessary
    target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
    if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
        im_res = target_res
        im, aff = edit_volumes.resample_volume(im, aff, im_res)
        if on_resample is not None:
            on_resample(im, aff)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    if ct:
        im = np.clip(im, 0, 80)
    im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    min_pad = cast(list[int], utils.reformat_to_list(min_pad, length=n_dims, dtype='int'))
    min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
    pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = utils.add_axis(im, axis=[0, -1])

    return im, aff, im_res, shape, pad_idx, crop_idx


def postprocess(post_patch_seg: RealArray | None, post_patch_parc: RealArray | None, shape: Sequence[int], pad_idx: IndexArray, crop_idx: IndexArray | None,
                labels_segmentation: IndexArray, labels_parcellation: IndexArray, aff: DoubleArray, im_res: RealArray, fast: bool, topology_classes: IndexArray | None, v1: bool,
                reuse_segmentation: bool=False, parcellation_postprocessor: ParcellationPostprocessor | None=None,
                segmentation_postprocessor: SegmentationPostprocessor | None=None, return_posteriors: bool=True, volume_dtype: npt.DTypeLike | None=np.float64) -> tuple[LabelArray, RealArray | None, RealArray]:
    """Convert network posteriors into labels, posteriors and soft volumes.

    Keeps the largest connected brain component, applies per-topological-class cleanup
    unless ``fast`` (or ``topology_classes`` is ``None``), takes the argmax, optionally
    parcellates cortex labels 3 and 42, pastes cropped patches back, and re-orients the
    result to ``aff``. Returns ``(seg, posteriors, volumes)``: an int32 label map,
    posteriors in the same orientation, and volumes in mm^3 rounded to three decimals
    (total intracranial first unless ``v1``, then non-background labels, then left and
    right parcels scaled to the hemisphere cortex volumes). ``volume_dtype`` is the
    accumulation dtype of the sums; ``None`` keeps the posteriors' dtype.

    ``post_patch_seg`` and ``post_patch_parc`` are the padded-grid posteriors; the NumPy
    parcellation path modifies ``post_patch_parc`` in place, as upstream does, and
    ``reuse_segmentation`` lets the segmentation posteriors be overwritten.
    ``shape``, ``pad_idx`` and ``crop_idx`` come from ``preprocess``; ``im_res`` is the
    voxel size in mm. See ``synthseg_mlx.mlx_engine.postprocess_arrays`` for the
    copy-safe wrapper. Parcel volume normalization can produce NaNs when a hemisphere's
    parcellation posterior sum is zero; the end-to-end runner rejects non-finite tables.
    """

    volumes = None
    if segmentation_postprocessor is not None:
        seg_patch, post_patch_seg, volumes = segmentation_postprocessor(
            cast(RealArray, post_patch_seg), pad_idx, labels_segmentation, fast, topology_classes,
            return_posteriors, component_filter=largest_component)
    else:
        # get posteriors
        if not isinstance(cast(object, post_patch_seg), np.ndarray):
            raise TypeError("post_patch_seg must be a NumPy array")
        post_patch_seg = np.squeeze(cast(RealArray, post_patch_seg))
        if fast | (topology_classes is None):
            post_patch_seg = edit_volumes.crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)

        # keep biggest connected component
        brain_mask = largest_component(np.sum(post_patch_seg[..., 1:], axis=-1) > 0.25)

        # reset posteriors to zero outside the largest connected component of each topological class
        if (not fast) & (topology_classes is not None):
            post_patch_seg_mask = post_patch_seg > 0.25
            post_patch_seg_mask[..., 1:] &= brain_mask[..., None]
            classes = np.unique(cast(IndexArray, topology_classes))
            post_patch_seg_mask[..., topology_classes == classes[0]] = True
            for topology_class in classes[1:]:
                tmp_topology_indices = np.where(topology_classes == topology_class)[0]
                tmp_mask = np.any(post_patch_seg_mask[..., tmp_topology_indices], axis=-1)
                tmp_mask = largest_component(tmp_mask)
                post_patch_seg_mask[..., tmp_topology_indices] = tmp_mask[..., None]
        else:
            post_patch_seg_mask = post_patch_seg > 0.2
            post_patch_seg_mask[..., 0] = True

        # Labels without topology cleanup still obey the brain mask.
        post_patch_seg_mask[..., 1:] &= brain_mask[..., None]
        # Owned host storage can be reused after inference.
        post_patch_seg = cast(RealArray, np.multiply(post_patch_seg, post_patch_seg_mask,
                                     out=post_patch_seg if reuse_segmentation else None))
        if (not fast) & (topology_classes is not None):
            post_patch_seg = edit_volumes.crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)

        # get hard segmentation
        post_patch_seg /= np.sum(post_patch_seg, axis=-1)[..., np.newaxis]
        seg_patch = labels_segmentation[argmax_channels(cast(RealArray, post_patch_seg))].astype('int32')

    # postprocess parcellation
    volumes_parc = None
    if post_patch_parc is not None:
        mask = (seg_patch == 3) | (seg_patch == 42)
        if parcellation_postprocessor is not None:
            seg_patch[mask], volumes_parc = parcellation_postprocessor(
                post_patch_parc, mask, pad_idx, labels_parcellation)
        else:
            if not isinstance(cast(object, post_patch_parc), np.ndarray):
                raise TypeError("post_patch_parc must be a NumPy array")
            post_patch_parc = np.squeeze(post_patch_parc)
            post_patch_parc = edit_volumes.crop_volume_with_idx(post_patch_parc, pad_idx, n_dims=3, return_copy=False)
            post_patch_parc[..., 0] = (~mask).astype(post_patch_parc.dtype)
            post_patch_parc /= np.sum(post_patch_parc, axis=-1)[..., np.newaxis]
            parcellation = cast(RealArray, post_patch_parc)
            seg_patch[mask] = labels_parcellation[np.argmax(parcellation[mask], axis=-1)]
            volumes_parc = np.sum(parcellation[..., 1:], axis=(0, 1, 2), dtype=volume_dtype)

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
        posteriors = None
        if cast(RealArray | None, post_patch_seg) is not None:
            posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
            posteriors[..., 0] = 1
            posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = post_patch_seg
    else:
        seg = seg_patch
        posteriors = post_patch_seg

    # align prediction back to first orientation
    seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    if cast(RealArray | None, posteriors) is not None:
        posteriors = edit_volumes.align_volume_to_ref(cast(RealArray, posteriors), np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)

    # Accumulate millions of voxel probabilities without float32 rounding drift.
    if volumes is None:
        volumes = np.sum(cast(RealArray, posteriors)[..., 1:], axis=tuple(range(0, len(cast(RealArray, posteriors).shape) - 1)), dtype=volume_dtype)
    total_volume_cortex_left = np.sum(volumes[np.where(labels_segmentation == 3)[0] - 1])
    total_volume_cortex_right = np.sum(volumes[np.where(labels_segmentation == 42)[0] - 1])
    if not v1:
        volumes = np.concatenate([np.array([np.sum(volumes)]), volumes])
    if volumes_parc is not None:
        volumes_parc_left = volumes_parc[:int(len(volumes_parc) / 2)]
        volumes_parc_right = volumes_parc[int(len(volumes_parc) / 2):]
        volumes_parc_left = volumes_parc_left / np.sum(volumes_parc_left) * total_volume_cortex_left
        volumes_parc_right = volumes_parc_right / np.sum(volumes_parc_right) * total_volume_cortex_right
        volumes = np.concatenate([volumes, volumes_parc_left, volumes_parc_right])
    volumes = cast(RealArray, np.around(volumes * np.prod(im_res), 3))

    return seg, posteriors, volumes


def largest_component(mask: BoolArray) -> BoolArray:
    """Return the largest connected component of ``mask`` (all False if ``mask`` is empty)."""
    # Empty margins cannot affect connectivity or component scan order.
    region = find_objects(mask, max_label=1)[0]
    result = np.zeros_like(mask)
    if region is not None:
        result[region] = edit_volumes.get_largest_connected_component(mask[region])
    return result


def argmax_channels(posteriors: RealArray) -> LabelArray:
    """Return the channel index of the maximum along the last axis, as int32."""
    # Cropped arrays have row gaps. Bound NumPy's contiguous temporary to one slab.
    labels = np.empty(posteriors.shape[:-1], dtype='int32')
    for index in range(posteriors.shape[0]):
        labels[index] = np.argmax(posteriors[index], axis=-1)
    return labels
