"""Contract tests for the MLX graph: runtime dependencies, layouts and QC cropping/scoring."""

import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from medmlx_core.errors import ModelExecutionError

from synthseg_mlx import BatchNorm3d, QCNet, SynthSegModel, crop_for_qc
from synthseg_mlx.mlx_model import repeat_upsample


def test_runtime_runs_without_training_frameworks() -> None:
    """Check native inference imports and runs without training frameworks."""
    program = """
import importlib.abc
import sys

class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'monai', 'tensorflow', 'keras', 'tf_keras', 'h5py'}:
            raise AssertionError(f'Forbidden runtime dependency: {fullname}')

sys.meta_path.insert(0, BlockImports())
import numpy as np
from synthseg_mlx import load_label_data, preprocess, postprocess_arrays
image, affine, resolution, shape, pad_idx, crop_idx = preprocess(
    np.arange(11 * 13 * 15, dtype=np.float64).reshape(11, 13, 15),
    np.eye(4), np.ones(3), ct=False, min_pad=32,
)
assert image.shape == (1, 32, 32, 32, 1)
assert image.min() == 0 and image.max() == 1
posterior = np.zeros((1, 32, 32, 32, 3), dtype=np.float32)
posterior[..., 0] = 0.1
posterior[..., 1] = 0.7
posterior[..., 2] = 0.2
seg, probabilities, volumes = postprocess_arrays(
    posterior, None, shape=shape, pad_idx=pad_idx, crop_idx=crop_idx,
    labels_segmentation=np.array([0, 3, 42]), labels_parcellation=np.array([0, 1001, 2001]),
    affine=affine, image_resolution=resolution, topology_classes=np.array([0, 1, 1]),
)
assert np.array_equal(seg, np.full((11, 13, 15), 3, dtype=np.int32))
assert volumes.dtype == np.float64
assert load_label_data()['segmentation'].shape == (55,)
assert not {'torch', 'monai', 'tensorflow', 'keras', 'tf_keras', 'h5py'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_invalid_layouts_fail_with_shared_core_error() -> None:
    """Reject invalid batch-normalization tensor layouts."""
    with pytest.raises(ModelExecutionError, match="batch normalization expects NDHWC"):
        BatchNorm3d(3, eps=1e-3)(mx.zeros((1, 2, 2, 3)))
    with pytest.raises(ModelExecutionError, match="cannot reproduce a source decoder resize"):
        repeat_upsample(mx.zeros((1, 2, 2, 2, 1)), mx.zeros((1, 3, 4, 4, 1)))
    with pytest.raises(ModelExecutionError, match="one 3D segmentation-index map"):
        crop_for_qc(mx.zeros((2, 2)))
    with pytest.raises(ModelExecutionError, match="requires contiguous QC class indices"):
        SynthSegModel(np.array([0, 3]), np.array([0, 1001]), np.array([0, 2]), np.array([0, 1]))


@pytest.mark.parametrize("channels", [8, 10, 16])
def test_qc_rejects_input_channels_that_do_not_match_checkpoint(channels: int) -> None:
    """Reject QC inputs with an incompatible number of channels."""
    with pytest.raises(ValueError, match="input channels"):
        QCNet(9, features=1)(mx.zeros((1, 16, 16, 16, channels)))


@pytest.mark.parametrize("invalid", ["rank", "dtype"])
def test_qc_class_map_requires_int32_ndhw(invalid: str) -> None:
    """Reject QC class maps with an invalid rank or dtype."""
    shape = (16, 16, 16) if invalid == "rank" else (1, 16, 16, 16)
    dtype = mx.int32 if invalid == "rank" else mx.float32
    with pytest.raises(ModelExecutionError, match="QC class indices require int32 NDHW"):
        QCNet(9, features=1).score_indices(mx.zeros(shape, dtype=dtype))
