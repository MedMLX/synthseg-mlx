"""Consumed APIs of the pinned SynthSeg definitions and TF/Keras recorder.

These protocols describe executed upstream definitions at the pinned
namespace boundary. They do not implement or replace the reference computations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

type HostArray = npt.NDArray[np.generic]
type RealArray = npt.NDArray[np.floating | np.integer | np.bool_]
type FloatArray = npt.NDArray[np.float32]
type DoubleArray = npt.NDArray[np.float64]
type IndexArray = npt.NDArray[np.int32 | np.int64]
type NameArray = npt.NDArray[np.str_]
type ArchiveValue = HostArray | Sequence[int] | int | bool


class Tensor(Protocol):
    """Describe a source tensor by name, shape and float32 host values."""

    name: str
    shape: Sequence[int]

    def numpy(self) -> FloatArray:
        """Return the reference tensor as a float32 host array."""
        ...


class Layer(Protocol):
    """Describe a source layer with its input tensor and named weights."""

    name: str
    input: Tensor
    weights: Sequence[Tensor]

    def set_weights(self, weights: Sequence[FloatArray]) -> None:
        """Set the reference layer weight arrays."""
        ...


class HasLayers(Protocol):
    """Describe an object that exposes its source layers."""

    layers: Sequence[Layer]


class ReferenceModel(HasLayers, Protocol):
    """Run a source graph and expose its inputs, layers and named layers."""

    inputs: Sequence[Tensor]

    def __call__(self, inputs: Sequence[RealArray], *, training: bool) -> list[Tensor]:
        """Run the reference graph and return its output tensors."""
        ...
    def get_layer(self, name: str) -> Layer:
        """Return a named reference model layer."""
        ...


class IntermediateModel(Protocol):
    """Run a source subgraph and return its output tensor."""

    def __call__(self, inputs: Sequence[RealArray], *, training: bool) -> Tensor:
        """Run the reference subgraph and return its tensor."""
        ...


class WeightLoader(Protocol):
    """Load source checkpoint weights into a reference model."""

    def __call__(self, model: ReferenceModel, path: str, **kwargs: object) -> ReferenceModel:
        """Load source weights into a reference model."""
        ...


class ModelFactory(Protocol):
    """Construct an intermediate source model and expose its weight loader."""

    load_weights: WeightLoader

    def __call__(self, inputs: Sequence[Tensor], outputs: Tensor) -> IntermediateModel:
        """Construct an intermediate reference model from tensors."""
        ...


class NetworkOptions(TypedDict):
    """Describe the keyword options accepted by source network builders."""

    nb_features: int
    input_shape: list[int | None]
    nb_levels: int
    conv_size: int
    name: str
    feat_mult: NotRequired[int]
    pool_size: NotRequired[int]
    padding: NotRequired[str]
    activation: NotRequired[str]
    nb_conv_per_level: NotRequired[int]
    batch_norm: NotRequired[int]
    input_model: NotRequired[ReferenceModel]
    nb_labels: NotRequired[int]
    use_logp: NotRequired[bool]
    final_pred_activation: NotRequired[str]


class SelectedLabels(TypedDict):
    """Describe the label assets selected for one recorded processing variant."""

    labels_segmentation: IndexArray
    flip_indices: IndexArray | None
    labels_denoiser: IndexArray
    labels_parcellation: IndexArray
    labels_qc: IndexArray
    topology_classes: IndexArray


class GraphLabels(TypedDict):
    """Describe the label assets used to construct the reduced reference graph."""

    labels_segmentation: IndexArray
    flip_indices: IndexArray | None
    labels_denoiser: IndexArray
    labels_parcellation: IndexArray
    labels_qc: IndexArray


class Geometry(TypedDict):
    """Describe the spatial geometry forwarded to source postprocessing."""

    shape: list[int]
    pad_idx: IndexArray
    crop_idx: IndexArray | None
    aff: DoubleArray
    im_res: RealArray


class Header(Protocol):
    """Read source image header fields by name."""

    def __getitem__(self, name: str) -> HostArray:
        """Read a named source image header field."""
        ...


class Volume(Protocol):
    """Describe a source NIfTI volume by its affine and voxel data."""

    affine: DoubleArray
    dataobj: npt.ArrayLike


class NiftiApi(Protocol):
    """Load, construct and save source NIfTI volumes."""

    def load(self, filename: str | Path) -> Volume:
        """Load a reference NIfTI volume."""
        ...
    def save(self, img: Volume, filename: str | Path) -> None:
        """Save a reference NIfTI volume."""
        ...
    def Nifti1Image(self, data: RealArray, affine: DoubleArray) -> Volume:
        """Construct a reference NIfTI image from voxels and affine."""
        ...


class UtilityNamespace(Protocol):
    """Inspect source volume data, geometry and header."""

    def get_volume_info(
        self, path: str, return_volume: bool
    ) -> tuple[RealArray, list[int], DoubleArray, int, int, Header, RealArray]:
        """Read source volume data and geometry for reference processing."""
        ...


class VersionModule(Protocol):
    """Expose the source module version string."""

    __version__: str


class TensorModule(VersionModule, Protocol):
    """Create source constant tensors and expose the module version."""

    def constant(self, value: RealArray | list[list[int]]) -> Tensor:
        """Create a source constant tensor."""
        ...


class ShapeOperation(Protocol):
    """Apply a source shape operation to input tensors."""

    def __call__(self, values: Sequence[Tensor]) -> Tensor:
        """Apply the source shape operation to tensors."""
        ...


class LayerTypes(Protocol):
    """Expose source layer classes used to identify graph layers."""

    Layer: type[object]
    Conv3D: type[object]
    BatchNormalization: type[object]


class ReferenceNamespace(Protocol):
    """Expose the pinned source pipeline functions, models and namespaces."""

    Model: ModelFactory
    KL: LayerTypes
    tf: TensorModule
    keras: VersionModule
    nib: NiftiApi
    utils: UtilityNamespace

    def get_flip_indices(
        self, labels: IndexArray, n_neutral: int
    ) -> tuple[IndexArray, IndexArray, IndexArray]:
        """Return source segmentation labels and left-right flip indices."""
        ...
    def preprocess(
        self,
        path_image: str,
        ct: bool,
        target_res: float | RealArray = 1.0,
        n_levels: int = 5,
        crop: list[int] | None = None,
        min_pad: int | list[int] | None = None,
        path_resample: str | None = None,
    ) -> tuple[
        RealArray, DoubleArray, Header, RealArray, list[int], IndexArray, IndexArray | None
    ]:
        """Run source image preparation and return its geometry."""
        ...
    def build_model(
        self,
        path_model_segmentation: str,
        path_model_parcellation: str,
        path_model_qc: str,
        input_shape_qc: int,
        labels_segmentation: IndexArray,
        labels_denoiser: IndexArray,
        labels_parcellation: IndexArray,
        labels_qc: IndexArray,
        sigma_smoothing: float,
        flip_indices: IndexArray | None,
        robust: bool,
        do_parcellation: bool,
        do_qc: bool,
    ) -> ReferenceModel:
        """Construct the requested reference SynthSeg model."""
        ...
    def postprocess(
        self,
        post_patch_seg: RealArray,
        post_patch_parc: RealArray | None,
        shape: list[int],
        pad_idx: IndexArray,
        crop_idx: IndexArray | None,
        labels_segmentation: IndexArray,
        labels_parcellation: IndexArray,
        aff: DoubleArray,
        im_res: RealArray,
        fast: bool,
        topology_classes: IndexArray | None,
        v1: bool,
    ) -> tuple[IndexArray, RealArray, RealArray]:
        """Run source segmentation and parcellation postprocessing."""
        ...
    def MakeShape(self, target: int) -> ShapeOperation:
        """Create the source shape operation for a target size."""
        ...


class DefinitionNamespace(Protocol):
    """Expose source definitions compiled from the pinned revision."""

    volshape_to_meshgrid: object
    get_flip_indices: object
    write_csv: object
    unet: NetworkBuilder
    conv_enc: NetworkBuilder


class NetworkBuilder(Protocol):
    """Construct a source neural network from keyword options."""

    def __call__(self, **kwargs: Unpack[NetworkOptions]) -> ReferenceModel:
        """Construct a source neural network using the supplied options."""
        ...


class ThreadConfig(Protocol):
    """Configure source tensor thread parallelism."""

    def set_intra_op_parallelism_threads(self, count: int) -> None:
        """Set source tensor intra-operation thread count."""
        ...
    def set_inter_op_parallelism_threads(self, count: int) -> None:
        """Set source tensor inter-operation thread count."""
        ...


class TensorConfig(Protocol):
    """Configure visible source devices and thread parallelism."""

    threading: ThreadConfig

    def set_visible_devices(self, devices: Sequence[object], device_type: str) -> None:
        """Set the source runtime devices visible to a device type."""
        ...


class TensorBindings(TensorModule, Protocol):
    """Describe the source tensor module with its runtime configuration."""

    config: TensorConfig


class ModelBindings(Protocol):
    """Expose the source model constructor."""

    Model: ModelFactory


class KerasBindings(VersionModule, Protocol):
    """Describe the source Keras module by its layers, backend and models."""

    layers: LayerTypes
    backend: object
    models: ModelBindings


class ArchiveWriter(Protocol):
    """Write named arrays to a reference fixture archive."""

    def __call__(self, file: Path, **arrays: ArchiveValue) -> None:
        """Write named arrays to a reference archive."""
        ...
