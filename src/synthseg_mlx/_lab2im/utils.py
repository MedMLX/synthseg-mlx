"""
This is an extracted subset of the upstream utilities. The following inventory
describes the upstream module, not all functions present here.
This file contains all the utilities used in that project. They are classified in 5 categories:
1- loading/saving functions:
    -load_volume
    -save_volume
    -get_volume_info
    -get_list_labels
    -load_array_if_path
    -write_pickle
    -read_pickle
    -write_model_summary
2- reformatting functions
    -reformat_to_list
    -reformat_to_n_channels_array
3- path related functions
    -list_images_in_folder
    -list_files
    -list_subfolders
    -strip_extension
    -strip_suffix
    -mkdir
    -mkcmd
4- shape-related functions
    -get_dims
    -get_resample_shape
    -add_axis
    -get_padding_margin
5- build affine matrices/tensors
    -create_affine_transformation_matrix
    -sample_affine_transform
    -create_rotation_transform
    -create_shearing_transform
6- miscellaneous
    -infer
    -LoopInfo
    -get_mapping_lut
    -build_training_generator
    -find_closest_number_divisible_by_m
    -build_binary_structure
    -draw_value_from_distribution
    -build_exp


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

import os
from collections.abc import Sequence
from typing import Literal, cast, overload

import nibabel as nib_runtime
import numpy as np

from synthseg_mlx._typing import (
    DoubleArray,
    FormatInput,
    FormatValue,
    HostArray,
    ImageHeader,
    IndexArray,
    IntArray,
    IntegerConcatApi,
    LabelArray,
    NiftiApi,
    RealArray,
    Resolution,
    Scalar,
)

# Extracted subset; imports and package boundaries adapted for synthseg-mlx.



nib = cast(NiftiApi, nib_runtime)


@overload
def load_volume(path_volume: str, im_only: Literal[True] = True, squeeze: bool = True, dtype: str | None = None, aff_ref: DoubleArray | None = None) -> RealArray: ...
@overload
def load_volume(path_volume: str, im_only: Literal[False], squeeze: bool = True, dtype: str | None = None, aff_ref: DoubleArray | None = None) -> tuple[RealArray, DoubleArray, ImageHeader]: ...
@overload
def load_volume(path_volume: str, im_only: bool, squeeze: bool = True, dtype: str | None = None, aff_ref: DoubleArray | None = None) -> RealArray | tuple[RealArray, DoubleArray, ImageHeader]: ...

def load_volume(path_volume: str, im_only: bool=True, squeeze: bool=True, dtype: str | None=None, aff_ref: DoubleArray | None=None) -> RealArray | tuple[RealArray, DoubleArray, ImageHeader]:
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with an identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        # the import is done here to avoid import loops
        from synthseg_mlx._lab2im import edit_volumes
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = edit_volumes.align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume: RealArray, aff: RealArray | Literal["FS"] | None, header: ImageHeader | None, path: str, res: Resolution | None=None, dtype: str | None=None, n_dims: int | None=3) -> None:
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is 3;
    if None, n_dims is inferred.
    """

    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty = nib.Nifti1Image(volume, aff, header)
            nifty.set_data_dtype(dtype)
        else:
            nifty = nib.Nifti1Image(volume, aff, header)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = cast(Resolution, reformat_to_list(res, length=n_dims, dtype=None))
            nifty.header.set_zooms(cast(Sequence[float], res))
        nib.save(nifty, path)


@overload
def get_volume_info(path_volume: str, return_volume: Literal[True], aff_ref: DoubleArray | None = None, max_channels: int = 10) -> tuple[RealArray, list[int] | IntArray, DoubleArray, int, int, ImageHeader, RealArray]: ...
@overload
def get_volume_info(path_volume: str, return_volume: Literal[False] = False, aff_ref: DoubleArray | None = None, max_channels: int = 10) -> tuple[list[int] | IntArray, DoubleArray, int, int, ImageHeader, RealArray]: ...

def get_volume_info(path_volume: str, return_volume: bool=False, aff_ref: DoubleArray | None=None, max_channels: int=10) -> tuple[RealArray, list[int] | IntArray, DoubleArray, int, int, ImageHeader, RealArray] | tuple[list[int] | IntArray, DoubleArray, int, int, ImageHeader, RealArray]:
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :param max_channels: maximum possible number of channels for the input volume.
    :return: volume (if return_volume is true), and corresponding info. If aff_ref is not None, the returned aff is
    the original one, i.e. the affine of the image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii' in path_volume:
        data_res = cast(RealArray, np.array(header['pixdim'][1:n_dims + 1]))
    elif '.mgz' in path_volume:
        data_res = cast(RealArray, np.array(header['delta']))  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        # the import is done here to avoid import loops
        from synthseg_mlx._lab2im import edit_volumes
        ras_axes = edit_volumes.get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = edit_volumes.get_ras_axes(aff_ref, n_dims=n_dims)
        im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


@overload
def get_list_labels(label_list: str | IndexArray | Sequence[int] | None = None, labels_dir: str | None = None, save_label_list: str | None = None, *, FS_sort: Literal[True]) -> tuple[LabelArray, int]: ...
@overload
def get_list_labels(label_list: str | IndexArray | Sequence[int] | None = None, labels_dir: str | None = None, save_label_list: str | None = None, FS_sort: Literal[False] = False) -> tuple[LabelArray, None]: ...
@overload
def get_list_labels(label_list: str | IndexArray | Sequence[int] | None = None, labels_dir: str | None = None, save_label_list: str | None = None, FS_sort: bool = False) -> tuple[LabelArray, int | None]: ...

def get_list_labels(label_list: str | IndexArray | Sequence[int] | None=None, labels_dir: str | None=None, save_label_list: str | None=None, FS_sort: bool=False) -> tuple[LabelArray, int | None]:
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param label_list: (optional) already computed label_list. Can be a sequence, a 1d numpy array, or the path to
    a numpy 1d array.
    :param labels_dir: Retained for source compatibility; folder and file discovery are not included in this subset.
    A label_list must be supplied.
    :param save_label_list: (optional) path where to save the label list.
    :param FS_sort: (optional) whether to sort label values according to the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided) labels if FS_sort is True.
    If one side of the brain is not represented at all in label_list, all labels are considered as neutral, and
    n_neutral_labels = len(label_list).
    """

    # load label list if previously computed
    if label_list is not None:
        label_list = cast(IndexArray, np.array(reformat_to_list(label_list, load_as_numpy=True, dtype='int')))

    # The labels_dir branch (reading every label map in a folder) is not extracted.
    else:
        raise Exception('either label_list, path_label_list or labels_dir should be provided')

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                             109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                             251, 252, 253, 254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                             502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 530,
                             531, 532, 533, 534, 535, 536, 537]
        neutral: list[int | np.int32 | np.int64] = list()
        left: list[int | np.int32 | np.int64] = list()
        right: list[int | np.int32 | np.int64] = list()
        for la in label_list:
            if la in neutral_FS_labels:
                if la not in neutral:
                    neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) | (1000 <= la <= 1035) | \
                    (la == 865) | (20100 < la < 20110):
                if la not in left:
                    left.append(la)
            elif (39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) | (20000 < la < 20010) | (la == 139) | \
                    (la == 866):
                if la not in right:
                    right.append(la)
            else:
                raise Exception('label {} not in our current FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = cast(IntegerConcatApi, np).concatenate([sorted(neutral), sorted(left), sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))

    if FS_sort:
        return cast(LabelArray, np.int32(label_list)), n_neutral_labels
    else:
        return cast(LabelArray, np.int32(label_list)), None


@overload
def load_array_if_path[T: HostArray](var: T, load_as_numpy: bool = True) -> T: ...
@overload
def load_array_if_path(var: FormatValue, load_as_numpy: bool = True) -> FormatValue: ...

def load_array_if_path(var: FormatValue, load_as_numpy: bool=True) -> FormatValue:
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(cast(str, var)), 'No such path: %s' % var
        var = cast(HostArray, np.load(cast(str, var)))
    return var


@overload
def reformat_to_list(var: None, length: int | None = None, load_as_numpy: bool = False, dtype: Literal["int", "float", "bool", "str"] | None = None) -> None: ...
@overload
def reformat_to_list(var: FormatInput, length: int | None = None, load_as_numpy: bool = False, *, dtype: Literal["int"]) -> list[int]: ...
@overload
def reformat_to_list(var: FormatInput, length: int | None = None, load_as_numpy: bool = False, *, dtype: Literal["float"]) -> list[float]: ...
@overload
def reformat_to_list(var: FormatInput, length: int | None = None, load_as_numpy: bool = False, *, dtype: Literal["bool"]) -> list[bool]: ...
@overload
def reformat_to_list(var: FormatInput, length: int | None = None, load_as_numpy: bool = False, *, dtype: Literal["str"]) -> list[str]: ...
@overload
def reformat_to_list(var: FormatInput, length: int | None = None, load_as_numpy: bool = False, dtype: None = None) -> list[Scalar]: ...
@overload
def reformat_to_list(var: FormatValue, length: int | None = None, load_as_numpy: bool = False, dtype: Literal["int", "float", "bool", "str"] | None = None) -> list[Scalar] | list[int] | list[float] | list[bool] | list[str] | None: ...

def reformat_to_list(var: FormatValue, length: int | None=None, load_as_numpy: bool=False, dtype: Literal["int", "float", "bool", "str"] | None=None) -> list[Scalar] | list[int] | list[float] | list[bool] | list[str] | None:
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, cast(tuple[type[object], ...], (int, float, np.int32, np.int64, np.float32, np.float64))):
        var = [cast(Scalar, var)]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = cast(list[Scalar], np.squeeze(var).tolist())
    elif isinstance(var, str):
        var = [cast(Scalar, var)]
    elif isinstance(cast(object, var), bool):
        var = [cast(Scalar, var)]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return cast(list[Scalar] | list[int] | list[float] | list[bool] | list[str] | None, var)


@overload
def reformat_to_n_channels_array(var: None, n_dims: int = 3, n_channels: int = 1) -> list[None]: ...
@overload
def reformat_to_n_channels_array(var: FormatInput, n_dims: int = 3, n_channels: int = 1) -> RealArray: ...
@overload
def reformat_to_n_channels_array(var: FormatValue, n_dims: int = 3, n_channels: int = 1) -> RealArray | list[None]: ...

def reformat_to_n_channels_array(var: FormatValue, n_dims: int=3, n_channels: int=1) -> RealArray | list[None]:
    """This function takes an int, float, list or tuple and reformat it to an array of shape (n_channels, n_dims).
    If resolution is a str, it will be assumed to be the path of a numpy array.
    If resolution is a numpy array, it will be checked to have shape (n_channels, n_dims).
    If var is None, it returns a list of n_channels None values."""
    if var is None:
        return [None] * n_channels
    if isinstance(var, str):
        var = cast(HostArray, np.load(var))
    # convert to numpy array
    if isinstance(var, (int, float, list, tuple)):
        var = reformat_to_list(var, n_dims)
        var = np.tile(np.array(var), (n_channels, 1))
    # check shape if numpy array
    elif isinstance(var, np.ndarray):
        if n_channels == 1:
            var = var.reshape((1, n_dims))
        else:
            if np.squeeze(var).shape == (n_dims,):
                var = np.tile(var.reshape((1, n_dims)), (n_channels, 1))
            elif var.shape != (n_channels, n_dims):
                raise ValueError('if array, var should be {0} or {1}'.format((1, n_dims), (n_channels, n_dims)))
    else:
        raise TypeError('var should be int, float, list, tuple or ndarray')
    return np.round(cast(RealArray, var), 3)


def get_dims(shape: Sequence[int], max_channels: int=10) -> tuple[int, int]:
    """Get the number of dimensions and channels from the shape of an array.
    If the final size is at most max_channels (default 10), it is interpreted as a channel count;
    otherwise all axes are spatial and the channel count is one.
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 15 > 10"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def add_axis(x: RealArray, axis: int | list[int]=0) -> RealArray:
    """Add axis to a numpy array.
    :param x: input array
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    axis = cast(list[int], reformat_to_list(axis))
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


def find_closest_number_divisible_by_m(n: int, m: int, answer_type: str='lower') -> int:
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher' (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)


def mkdir(path_dir: str) -> None:
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)
