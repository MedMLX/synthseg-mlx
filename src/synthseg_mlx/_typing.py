"""Array, image and callback domains of the pinned SynthSeg pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from mlx.core import Dtype, StreamOrDevice
    from mlx.core import array as NativeArray
    from mlx.nn.layers.base import Module as NativeModule
    from mlx.nn.layers.convolution import Conv3d as NativeConv3d

type HostArray = npt.NDArray[np.generic]
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
    def Nifti2Image(
        self, dataobj: RealArray, affine: RealArray | str, header: ImageHeader | None = None
    ) -> ImageVolume:
        """Construct a NIfTI-2 image from data, affine, and optional header."""
        ...
    def save(self, img: ImageVolume, filename: str) -> None:
        """Write a NIfTI image to a path."""
        ...


class SegmentationPostprocessor(Protocol):
    def __call__(
        self,
        posteriors: RealArray,
        pad_idx: IndexArray,
        labels: IndexArray,
        fast: bool,
        topology: IndexArray | None,
        return_posteriors: bool,
        *,
        component_filter: Callable[[BoolArray], BoolArray],
    ) -> tuple[LabelArray, RealArray | None, RealArray]:
        """Return labels, optional posteriors, and volumes from segmentation probabilities."""
        ...


class ParcellationPostprocessor(Protocol):
    def __call__(
        self, posteriors: RealArray, mask: BoolArray, pad_idx: IndexArray, labels: IndexArray
    ) -> tuple[LabelArray, RealArray]:
        """Postprocess masked cortex probabilities into parcel labels and volumes."""
        ...


type ParameterValue = NativeArray | dict[str, ParameterValue] | list[ParameterValue]


class NativeCallable(Protocol):
    def __call__(self, values: NativeArray) -> NativeArray:
        """Apply a native neural-network operation to an array."""
        ...


class NativeOperation(NativeCallable, Protocol):
    weight: NativeArray
    bias: NativeArray


class NeuralRuntime(Protocol):
    Module: type[NativeModule]
    Conv3d: type[NativeConv3d]

    def MaxPool3d(self, kernel_size: int, stride: int | None = None) -> NativeCallable:
        """Construct a three-dimensional native max-pooling layer."""
        ...

    def relu(self, values: NativeArray) -> NativeArray:
        """Apply the rectified linear activation."""
        ...

    def elu(self, values: NativeArray, alpha: float = 1.0) -> NativeArray:
        """Apply the exponential linear activation."""
        ...


class NativeArrayApi(Protocol):
    def array(self, val: RealArray | NativeArray, dtype: Dtype | None = None) -> NativeArray:
        """Create an MLX array from host or native values."""
        ...
    def eval(self, *args: NativeArray | dict[str, ParameterValue]) -> None:
        """Evaluate arrays in native parameter trees."""
        ...


class NativeCheckpointApi(Protocol):
    @property
    def float32(self) -> Dtype:
        """Return the MLX float32 dtype."""
        ...
    def load(self, file: str, /, format: Literal["safetensors"]) -> dict[str, NativeArray]:
        """Load named arrays from a safetensors file."""
        ...
    def isfinite(self, value: NativeArray) -> NativeArray:
        """Check native array elements for finiteness."""
        ...
    def all(self, value: NativeArray) -> NativeArray:
        """Reduce native boolean values to a scalar."""
        ...
    def eval(self, *args: NativeArray) -> None:
        """Evaluate native arrays."""
        ...


class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: list[NativeArray],
        template: list[tuple[str, int]],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[Dtype],
        stream: StreamOrDevice,
    ) -> list[NativeArray]:
        """Dispatch the kernel with explicit inputs, launch geometry, and output shapes."""
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
