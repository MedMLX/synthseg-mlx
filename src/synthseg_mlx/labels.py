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

# Extracted label correspondence and CSV writer; explicit integer scalars support NumPy 2.
import csv
import os
from collections.abc import Sequence
from typing import cast

import numpy as np

from synthseg_mlx._lab2im import utils
from synthseg_mlx._typing import (
    IndexArray,
    IntArray,
    NameArray,
)


def get_flip_indices(
    labels_segmentation: IndexArray | Sequence[int], n_neutral_labels: int,
) -> tuple[IndexArray, IndexArray, IntArray]:
    """Return sorted unique labels, left/right flip indices and the unique indices.

    ``labels_segmentation`` is the source label list: ``n_neutral_labels`` left/right
    neutral entries, then the left structures, then their right counterparts in the same
    order. ``flip_indices[i]`` is the position (in the sorted unique list) of the
    contralateral twin of label ``i``, or ``i`` itself for neutral labels. ``unique_idx``
    are the positions of the unique labels in the input, used to select aligned arrays
    such as names, topology classes and QC labels.
    """

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondence between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0].item()

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])].item()
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])].item()

    return labels_segmentation, flip_indices, unique_idx


def write_csv(path_csv: str, data: Sequence[object] | None, unique_file: bool, labels: IndexArray, names: NameArray | None, skip_first: bool=True, last_first: bool=False) -> None:
    """Write or append to a CSV with one column per label.

    With ``unique_file`` and ``data=None`` the file is created holding only a ``subject``
    header built from the unique labels (their names when ``names`` is given); with
    ``unique_file`` and ``data`` (subject name, then one value per column) one row is
    appended. Without ``unique_file`` a two-row file (header, then ``data``) is written.
    ``skip_first`` drops the background column and ``last_first`` moves the last column
    (total intracranial volume) to the front. Raises ``ValueError`` if ``data`` is
    ``None`` for a non-unique file.
    """

    # initialisation
    utils.mkdir(os.path.dirname(path_csv))
    labels, unique_idx = np.unique(labels, return_index=True)
    if skip_first:
        labels = labels[1:]
    if names is not None:
        names_list = cast(list[str], names[unique_idx].tolist())
        if skip_first:
            names_list = names_list[1:]
        header = names_list
    else:
        header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')

    # modify data
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]

    # write csv
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(cast(Sequence[Sequence[object]], data))
