"""Native MLX implementation of SynthSeg.

Public names load on first access, so importing the package (or the medmlx
declaration in ``synthseg_mlx.medmlx``) does not import MLX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synthseg_mlx.label_data import load_label_data as load_label_data
    from synthseg_mlx.labels import get_flip_indices as get_flip_indices
    from synthseg_mlx.mlx_engine import postprocess_arrays as postprocess_arrays
    from synthseg_mlx.mlx_engine import predict_numpy as predict_numpy
    from synthseg_mlx.mlx_engine import prepare_network_input as prepare_network_input
    from synthseg_mlx.mlx_engine import qc_scores as qc_scores
    from synthseg_mlx.mlx_model import BatchNorm3d as BatchNorm3d
    from synthseg_mlx.mlx_model import CheckpointLayers as CheckpointLayers
    from synthseg_mlx.mlx_model import QCNet as QCNet
    from synthseg_mlx.mlx_model import SynthSegModel as SynthSegModel
    from synthseg_mlx.mlx_model import UNet as UNet
    from synthseg_mlx.mlx_model import crop_for_qc as crop_for_qc
    from synthseg_mlx.processing import postprocess as postprocess
    from synthseg_mlx.processing import preprocess as preprocess
    from synthseg_mlx.variants import SynthSegVariant as SynthSegVariant

__version__ = "0.3.3"

_EXPORTS = {
    "BatchNorm3d": "synthseg_mlx.mlx_model",
    "CheckpointLayers": "synthseg_mlx.mlx_model",
    "QCNet": "synthseg_mlx.mlx_model",
    "SynthSegModel": "synthseg_mlx.mlx_model",
    "SynthSegVariant": "synthseg_mlx.variants",
    "UNet": "synthseg_mlx.mlx_model",
    "crop_for_qc": "synthseg_mlx.mlx_model",
    "get_flip_indices": "synthseg_mlx.labels",
    "load_label_data": "synthseg_mlx.label_data",
    "postprocess": "synthseg_mlx.processing",
    "postprocess_arrays": "synthseg_mlx.mlx_engine",
    "predict_numpy": "synthseg_mlx.mlx_engine",
    "prepare_network_input": "synthseg_mlx.mlx_engine",
    "preprocess": "synthseg_mlx.processing",
    "qc_scores": "synthseg_mlx.mlx_engine",
}

__all__ = [
    "BatchNorm3d",
    "CheckpointLayers",
    "QCNet",
    "SynthSegModel",
    "SynthSegVariant",
    "UNet",
    "crop_for_qc",
    "get_flip_indices",
    "load_label_data",
    "postprocess",
    "postprocess_arrays",
    "predict_numpy",
    "prepare_network_input",
    "preprocess",
    "qc_scores",
]


def __getattr__(name: str) -> object:
    """Load a public export from its module on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'synthseg_mlx' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List eager and lazy package exports."""
    return sorted({*globals(), *_EXPORTS})
