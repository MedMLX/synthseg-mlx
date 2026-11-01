"""
This is an extracted subset of the upstream volume helpers. The following
inventory describes the upstream module, not all functions present here.
This file contains functions to edit/preprocess volumes (i.e. not tensors!).
These functions are sorted in five categories:
1- volume editing: this can be applied to any volume (i.e. images or label maps). It contains:
        -mask_volume
        -rescale_volume
        -crop_volume
        -crop_volume_around_region
        -crop_volume_with_idx
        -pad_volume
        -flip_volume
        -resample_volume
        -resample_volume_like
        -get_ras_axes
        -align_volume_to_ref
        -blur_volume
2- label map editing: can be applied to label maps only. It contains:
        -correct_label_map
        -mask_label_map
        -smooth_label_map
        -erode_label_map
        -get_largest_connected_component
        -compute_hard_volumes
        -compute_distance_map
3- editing all volumes in a folder: functions are more or less the same as 1, but they now apply to all the volumes
in a given folder. Thus we provide folder paths rather than numpy arrays as inputs. It contains:
        -mask_images_in_dir
        -rescale_images_in_dir
        -crop_images_in_dir
        -crop_images_around_region_in_dir
        -pad_images_in_dir
        -flip_images_in_dir
        -align_images_in_dir
        -correct_nans_images_in_dir
        -blur_images_in_dir
        -create_mutlimodal_images
        -convert_images_in_dir_to_nifty
        -mri_convert_images_in_dir
        -samseg_images_in_dir
        -niftyreg_images_in_dir
        -upsample_anisotropic_images
        -simulate_upsampled_anisotropic_images
        -check_images_in_dir
4- label maps in dir: same as 3 but for label map-specific functions. It contains:
        -correct_labels_in_dir
        -mask_labels_in_dir
        -smooth_labels_in_dir
        -erode_labels_in_dir
        -upsample_labels_in_dir
        -compute_hard_volumes_in_dir
        -build_atlas
5- dataset editing: functions for editing datasets (i.e. images with corresponding label maps). It contains:
        -check_images_and_labels
        -crop_dataset_to_minimum_size
        -subdivide_dataset_to_patches


If you use this code, please cite the first SynthSeg paper:
https://github.com/BBillot/lab2im/blob/master/bibtex.bib

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

from typing import Literal, cast, overload

# Extracted subset; imports and package boundaries adapted for synthseg-mlx.
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import label as scipy_label

from synthseg_mlx._lab2im import utils
from synthseg_mlx._typing import (
    BoolArray,
    DoubleArray,
    FloatGridApi,
    IndexArray,
    IntArray,
    IntegerGridApi,
    NumericScalar,
    RealArray,
    Resolution,
    Size,
)


def rescale_volume(volume: RealArray, new_min: float=0, new_max: float=255, min_percentile: float=2, max_percentile: float=98, use_positive_only: bool=False) -> RealArray:
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = cast(NumericScalar, np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile))
    robust_max = cast(NumericScalar, np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile))

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return cast(RealArray, new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min))
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


@overload
def crop_volume[V: RealArray](volume: V, cropping_margin: Size | None = None, cropping_shape: Size | None = None, aff: None = None, return_crop_idx: Literal[False] = False, mode: str = "center") -> V: ...
@overload
def crop_volume[V: RealArray](volume: V, cropping_margin: Size | None = None, cropping_shape: Size | None = None, aff: None = None, *, return_crop_idx: Literal[True], mode: str = "center") -> tuple[V, IndexArray]: ...
@overload
def crop_volume[V: RealArray](volume: V, cropping_margin: Size | None = None, cropping_shape: Size | None = None, *, aff: DoubleArray, return_crop_idx: Literal[False] = False, mode: str = "center") -> tuple[V, DoubleArray]: ...
@overload
def crop_volume[V: RealArray](volume: V, cropping_margin: Size | None = None, cropping_shape: Size | None = None, *, aff: DoubleArray, return_crop_idx: Literal[True], mode: str = "center") -> tuple[V, DoubleArray, IndexArray]: ...
@overload
def crop_volume[V: RealArray](volume: V, cropping_margin: Size | None = None, cropping_shape: Size | None = None, aff: DoubleArray | None = None, return_crop_idx: bool = False, mode: str = "center") -> V | tuple[RealArray, ...]: ...

def crop_volume(volume: RealArray, cropping_margin: Size | None=None, cropping_shape: Size | None=None, aff: DoubleArray | None=None, return_crop_idx: bool=False, mode: str='center') -> RealArray | tuple[RealArray, ...]:
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. The cropping margin is applied on both sides.
    Can be an int, sequence or 1d numpy array of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None, and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = utils.get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = cast(list[int], utils.reformat_to_list(cropping_margin, length=n_dims))
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = cast(list[int], utils.reformat_to_list(cropping_shape, length=n_dims))
        if mode == 'center':
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum([min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                                      np.array(vol_shape)[:n_dims])
        elif mode == 'random':
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)


@overload
def crop_volume_with_idx[V: RealArray](volume: V, crop_idx: IndexArray, aff: None = None, n_dims: int | None = None, return_copy: bool = True) -> V: ...
@overload
def crop_volume_with_idx[V: RealArray](volume: V, crop_idx: IndexArray, aff: DoubleArray, n_dims: int | None = None, return_copy: bool = True) -> tuple[V, DoubleArray]: ...
@overload
def crop_volume_with_idx[V: RealArray](volume: V, crop_idx: IndexArray, aff: DoubleArray | None = None, n_dims: int | None = None, return_copy: bool = True) -> V | tuple[V, DoubleArray]: ...

def crop_volume_with_idx(volume: RealArray, crop_idx: IndexArray, aff: DoubleArray | None=None, n_dims: int | None=None, return_copy: bool=True) -> RealArray | tuple[RealArray, RealArray]:
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: cropping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims is
    inferred as half the number of crop indices.
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume


@overload
def pad_volume[V: RealArray](volume: V, padding_shape: Size, padding_value: float = 0, aff: None = None, return_pad_idx: Literal[False] = False) -> V: ...
@overload
def pad_volume[V: RealArray](volume: V, padding_shape: Size, padding_value: float = 0, aff: None = None, *, return_pad_idx: Literal[True]) -> tuple[V, IndexArray]: ...
@overload
def pad_volume[V: RealArray](volume: V, padding_shape: Size, padding_value: float = 0, *, aff: DoubleArray, return_pad_idx: Literal[False] = False) -> tuple[V, DoubleArray]: ...
@overload
def pad_volume[V: RealArray](volume: V, padding_shape: Size, padding_value: float = 0, *, aff: DoubleArray, return_pad_idx: Literal[True]) -> tuple[V, DoubleArray, IndexArray]: ...
@overload
def pad_volume[V: RealArray](volume: V, padding_shape: Size, padding_value: float = 0, aff: DoubleArray | None = None, return_pad_idx: bool = False) -> V | tuple[RealArray, ...]: ...

def pad_volume(volume: RealArray, padding_shape: Size, padding_value: float=0, aff: DoubleArray | None=None, return_pad_idx: bool=False) -> RealArray | tuple[RealArray, ...]:
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :param return_pad_idx: (optional) the pad_idx corresponds to the indices where we should crop the resulting
    padded image (ie the output of this function) to go back to the original volume (ie the input of this function).
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = utils.get_dims(vol_shape)
    padding_shape = utils.reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)


def resample_volume(volume: RealArray, aff: DoubleArray, new_vox_size: Resolution, interpolation: str='linear', blur: bool=True) -> tuple[RealArray, DoubleArray]:
    """Resize voxels to a provided size and return an adjusted affine.
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :param interpolation: (optional) type of interpolation. Can be 'linear' or 'nearest'. Default is 'linear'.
    :param blur: (optional) whether to blur before resampling. Upsampled axes use zero
    blur; unchanged and downsampled axes use the source Gaussian rule. Default is True.
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas) if blur else volume

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = cast(FloatGridApi, np).arange(start=start[0], stop=stop[0], step=step[0])
    yi = cast(FloatGridApi, np).arange(start=start[1], stop=stop[1], step=step[1])
    zi = cast(FloatGridApi, np).arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    if interpolation == 'linear':
        # Keep the original grid, boundary clamping and float64 interpolation output.
        coordinates = np.array(np.broadcast_arrays(xig, yig, zig))
        volume2 = map_coordinates(volume_filt, coordinates, order=1, mode='nearest', prefilter=False,
                                  output=np.result_type(volume_filt.dtype, np.float64))
    else:
        axes = tuple(cast(IntegerGridApi, np).arange(size) for size in volume_filt.shape)
        interpolator = RegularGridInterpolator(axes, volume_filt, method=cast(Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"], interpolation))
        volume2 = interpolator((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def get_ras_axes(aff: DoubleArray, n_dims: int=3) -> IntArray:
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: invertible square affine matrix, with at least n_dims rows and columns.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: one numpy 1d array of length n_dims with the axes corresponding to RAS orientations.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


@overload
def align_volume_to_ref[V: RealArray](volume: V, aff: DoubleArray, aff_ref: DoubleArray | None = None, return_aff: Literal[False] = False, n_dims: int | None = None, return_copy: bool = True) -> V: ...
@overload
def align_volume_to_ref[V: RealArray](volume: V, aff: DoubleArray, aff_ref: DoubleArray | None = None, *, return_aff: Literal[True], n_dims: int | None = None, return_copy: bool = True) -> tuple[V, DoubleArray]: ...
@overload
def align_volume_to_ref[V: RealArray](volume: V, aff: DoubleArray, aff_ref: DoubleArray | None = None, return_aff: bool = False, n_dims: int | None = None, return_copy: bool = True) -> V | tuple[V, DoubleArray]: ...

def align_volume_to_ref(volume: RealArray, aff: DoubleArray, aff_ref: DoubleArray | None=None, return_aff: bool=False, n_dims: int | None=None, return_copy: bool=True) -> RealArray | tuple[RealArray, RealArray]:
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = utils.get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            # NumPy 2 requires an explicit scalar for this one-element assignment.
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx].item()

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


def get_largest_connected_component(mask: BoolArray, structure: BoolArray | None=None) -> BoolArray:
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()
