"""Array, image and callback domains of the pinned SynthSeg pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from medmlx_core.typing import HostArray

type RealArray = npt.NDArray[np.floating | np.integer | np.bool_]
type FloatArray = npt.NDArray[np.float32]
type DoubleArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type IndexArray = npt.NDArray[np.int32 | np.int64]
type LabelArray = npt.NDArray[np.int32]
type BoolArray = npt.NDArray[np.bool_]
type NameArray = npt.NDArray[np.str_]
type Scalar = int | float | bool | str | np.integer | np.floating
type NumericScalar = int | float | np.integer | np.floating
type FormatInput = Scalar | Sequence[Scalar] | HostArray
type FormatValue = FormatInput | None
type Size = int | Sequence[int] | IndexArray
type Resolution = float | Sequence[float] | RealArray


class LabelAssets(TypedDict):
    segmentation: LabelArray
    segmentation_names: NameArray
    topology_classes: LabelArray
    parcellation: LabelArray
    parcellation_names: NameArray
    qc: LabelArray
    qc_names: NameArray


class LabelData(LabelAssets):
    denoiser: NotRequired[LabelArray]


class RobustLabelData(LabelAssets):
    denoiser: LabelArray


class ImageHeader(Protocol):
    def __getitem__(self, key: str) -> HostArray:
        """Read a named image header field."""
        ...
    def __setitem__(self, key: str, value: RealArray | float | str) -> None:
        """Set a named image header field."""
        ...
    def set_zooms(self, zooms: Sequence[float]) -> None:
        """Set physical voxel spacing in the header."""
        ...
    def get_xyzt_units(self) -> tuple[str, str]:
        """Return the spatial and temporal unit names in the NIfTI header."""
        ...
    def set_xyzt_units(self, xyz: str, t: str) -> None:
        """Set the spatial and temporal unit names in the NIfTI header."""
        ...
    def copy(self) -> ImageHeader:
        """Return a copy of the image header."""
        ...


class ImageVolume(Protocol):
    @property
    def affine(self) -> DoubleArray:
        """Return the image voxel-to-world affine."""
        ...
    @property
    def header(self) -> ImageHeader:
        """Return the image header."""
        ...
    @property
    def shape(self) -> tuple[int, ...]:
        """Return image dimensions."""
        ...
    @property
    def dataobj(self) -> npt.ArrayLike:
        """Return the image voxel data proxy."""
        ...
    def get_data_dtype(self) -> np.dtype[np.generic]:
        """Return the stored image dtype."""
        ...
    def get_fdata(self) -> DoubleArray:
        """Return scaled image data as float64."""
        ...
    def set_data_dtype(self, datatype: str) -> None:
        """Set the output image data dtype."""
        ...


class NiftiApi(Protocol):
    def load(self, filename: str) -> ImageVolume:
        """Read a NIfTI image from a path."""
        ...
    def Nifti1Header(self) -> ImageHeader:
        """Create a new NIfTI header."""
        ...
    def Nifti1Image(
        self, dataobj: RealArray, affine: RealArray | str, header: ImageHeader | None = None
    ) -> ImageVolume:
        """Construct a NIfTI image from data, affine, and optional header."""
        ...
    def save(self, img: ImageVolume, filename: str) -> None:
        """Write a NIfTI image to a path."""
        ...


class FloatGridApi(Protocol):
    def arange(
        self, *, start: NumericScalar, stop: NumericScalar, step: NumericScalar
    ) -> DoubleArray:
        """Construct the float64 grid over the requested interval."""
        ...


class IntegerConcatApi(Protocol):
    def concatenate(self, arrays: Sequence[Sequence[int | np.int32 | np.int64]]) -> IndexArray:
        """Concatenate integer sequences into an index array."""
        ...


class IntegerGridApi(Protocol):
    def arange(self, stop: int) -> IntArray:
        """Create integer indices from zero to the exclusive stop."""
        ...
