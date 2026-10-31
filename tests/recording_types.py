"""Array roles admitted by the version-two retained oracle recordings."""

from collections.abc import Iterable, Iterator
from typing import Literal, Protocol, overload

import numpy as np
import numpy.typing as npt

from synthseg_mlx._typing import (
    BoolArray,
    DoubleArray,
    FloatArray,
    HostArray,
    IndexArray,
    IntArray,
    LabelArray,
    NameArray,
    RealArray,
)


class Recording(Protocol):
    @overload
    def __getitem__(
        self,
        key: Literal[
            "denoiser_features",
            "expected_pad_idx",
            "expected_shape",
            "features",
            "min_pad",
            "n_levels",
            "pad_idx",
            "qc_shape",
            "seed",
            "shape",
            "target",
            "weight_shapes",
        ],
    ) -> IntArray:
        """Read integer shape, count, or seed data from the recording."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "asset_names",
            "nibabel",
            "numpy",
            "reference_device",
            "reference_host",
            "scipy",
            "source_paths",
            "synthseg_commit",
            "synthseg_parcellation_names",
            "synthseg_qc_names",
            "synthseg_qc_names_2.0",
            "synthseg_segmentation_names",
            "synthseg_segmentation_names_2.0",
            "tensorflow",
            "tf_keras",
            "volume_names",
            "weight_keys",
            "weight_layers",
        ],
    ) -> NameArray:
        """Read string names, provenance, or tensor keys from the recording."""
        ...
    @overload
    def __getitem__(
        self, key: Literal["ct", "fast", "has_parcellation", "robust", "v1"]
    ) -> BoolArray:
        """Read a boolean variant flag from the recording."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "crop",
            "crop_idx",
            "expected_crop_idx",
            "expected_posteriors",
            "expected_resolution",
            "input",
            "topology_classes",
        ],
    ) -> RealArray:
        """Read a numeric input, crop, or topology array."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "crop_runner_input",
            "crop_runner_qc",
            "crop_runner_resampled",
            "crop_runner_volumes",
            "ct_runner_input",
            "ct_runner_qc",
            "ct_runner_resampled",
            "image_resolution",
            "parcellation",
            "parcellation_input",
            "posteriors",
            "qc",
            "resampled_image",
            "runner_input",
            "runner_qc",
            "runner_resampled",
            "runner_volumes",
            "segmentation",
            "weight_boost_values",
            "weight_offsets",
            "weight_scales",
        ],
    ) -> FloatArray:
        """Read float32 inputs, predictions, or synthetic weight data."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "crop_runner_mask",
            "ct_runner_mask",
            "expected",
            "expected_mask",
            "flip_indices",
            "labels_denoiser",
            "labels_qc",
            "mask",
            "runner_mask",
            "synthseg_denoiser_labels_2.0",
            "synthseg_parcellation_labels",
            "synthseg_qc_labels",
            "synthseg_qc_labels_2.0",
            "synthseg_segmentation_labels",
            "synthseg_segmentation_labels_2.0",
            "synthseg_topological_classes",
            "synthseg_topological_classes_2.0",
            "volume_labels",
            "weight_boost_indices",
        ],
    ) -> LabelArray:
        """Read labels, masks, or stored integer indices."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "aff",
            "affine",
            "crop_runner_affine",
            "crop_runner_input_affine",
            "ct_runner_affine",
            "ct_runner_input_affine",
            "ct_runner_volumes",
            "expected_affine",
            "expected_image",
            "expected_volumes",
            "im_res",
            "resampled_affine",
            "runner_affine",
            "runner_input_affine",
            "target_res",
            "volumes",
        ],
    ) -> DoubleArray:
        """Read float64 geometry, image, or volume measurements."""
        ...
    @overload
    def __getitem__(self, key: Literal["weight_seeds"]) -> npt.NDArray[np.uint32]:
        """Read the recorded unsigned weight seeds."""
        ...
    @overload
    def __getitem__(
        self, key: Literal["labels_parcellation", "labels_segmentation"]
    ) -> IndexArray:
        """Read the selected segmentation or parcellation label indices."""
        ...
    @overload
    def __getitem__(self, key: str) -> HostArray:
        """Read a named host array from the recording."""
        ...
    def items(self) -> Iterable[tuple[str, HostArray]]:
        """Iterate stored array names and their host arrays."""
        ...
    def __iter__(self) -> Iterator[str]:
        """Iterate stored array names."""
        ...
