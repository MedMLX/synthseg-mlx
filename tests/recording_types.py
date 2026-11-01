"""Array roles of the retained upstream oracle recordings that the tests read by name."""

from collections.abc import Iterable
from typing import Literal, Protocol, overload

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
            "denoiser_features", "features", "min_pad", "n_levels", "pad_idx", "qc_shape", "shape"
        ],
    ) -> IntArray:
        """Read integer shape, count, or padding data."""
        ...
    @overload
    def __getitem__(
        self, key: Literal["reference_host", "synthseg_commit", "volume_names"]
    ) -> NameArray:
        """Read provenance or volume names."""
        ...
    @overload
    def __getitem__(
        self, key: Literal["ct", "fast", "has_parcellation", "robust", "v1"]
    ) -> BoolArray:
        """Read a boolean variant or recording flag."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal["crop", "crop_idx", "expected_posteriors", "input", "topology_classes"],
    ) -> RealArray:
        """Read a numeric input, crop, or topology array."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "crop_runner_input",
            "crop_runner_volumes",
            "ct_runner_input",
            "image_resolution",
            "parcellation",
            "parcellation_input",
            "qc",
            "runner_input",
            "runner_qc",
            "runner_volumes",
            "segmentation",
        ],
    ) -> FloatArray:
        """Read float32 inputs, predictions, or runner measurements."""
        ...
    @overload
    def __getitem__(
        self,
        key: Literal[
            "expected",
            "expected_mask",
            "flip_indices",
            "labels_denoiser",
            "labels_qc",
            "mask",
            "volume_labels",
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
            "crop_runner_input_affine",
            "ct_runner_input_affine",
            "expected_volumes",
            "im_res",
            "runner_input_affine",
            "target_res",
            "volumes",
        ],
    ) -> DoubleArray:
        """Read float64 geometry or volume measurements."""
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
