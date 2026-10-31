"""Native checkpoint regressions retained across the safetensors migration."""

from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import numpy as np
import pytest
from medmlx_core.errors import ModelExecutionError
from mlx.core import array as NativeArray

from synthseg_mlx import CheckpointLayers
from synthseg_mlx._typing import NativeArrayApi
from synthseg_mlx.mlx_model import RobustSegmentation, load_safetensors


class NativeCheckpointWriter(Protocol):
    def save_safetensors(self, file: str, arrays: dict[str, NativeArray]) -> None:
        """Write named native arrays to a safetensors file."""
        ...


class ModelCheckpointWriter(Protocol):
    def save_weights(self, file: str) -> None:
        """Write the model weights to a file."""
        ...


def test_native_robust_checkpoint_restores_all_three_networks(tmp_path: Path) -> None:
    """Restore and compare the coarse, denoiser, and fine network parameters."""
    source = RobustSegmentation(2, 3, features=1, denoiser_features=1)
    path = tmp_path / "synthseg_robust_2.0.safetensors"
    cast(ModelCheckpointWriter, source).save_weights(str(path))
    restored = RobustSegmentation(2, 3, features=1, denoiser_features=1)
    load_safetensors(restored, path)
    for original_network, restored_network, prefix in (
        (source.coarse, restored.coarse, "unet"),
        (source.denoiser, restored.denoiser, "l2l"),
        (source.fine, restored.fine, "unet2"),
    ):
        original_layer = original_network.layer(f"{prefix}_conv_downarm_0_0")
        restored_layer = restored_network.layer(f"{prefix}_conv_downarm_0_0")
        np.testing.assert_array_equal(
            np.asarray(restored_layer.weight, dtype=np.float32),
            np.asarray(original_layer.weight, dtype=np.float32),
        )


def test_native_weights_preserve_parameters_through_a_cache_symlink(tmp_path: Path) -> None:
    """Snapshot links keep the safetensors name while their targets have no suffix."""
    native = tmp_path / "native.safetensors"
    array_api = cast(NativeArrayApi, mx)
    tensors = {
        "test_conv.weight": array_api.array(
            np.arange(3 * 3 * 3 * 2 * 3, dtype=np.float32)
            .reshape(3, 3, 3, 2, 3)
            .transpose(4, 0, 1, 2, 3)
        ),
        "test_conv.bias": array_api.array(np.array([1, 2, 3], dtype=np.float32)),
        "test_bn.weight": array_api.array(np.array([2, 3, 4], dtype=np.float32)),
        "test_bn.bias": array_api.array(np.array([3, 4, 5], dtype=np.float32)),
        "test_bn.running_mean": array_api.array(np.array([4, 5, 6], dtype=np.float32)),
        "test_bn.running_var": array_api.array(np.array([5, 6, 7], dtype=np.float32)),
    }
    cast(NativeCheckpointWriter, mx).save_safetensors(str(native), tensors)
    blob = tmp_path / "blob-without-a-suffix"
    native.rename(blob)
    native.symlink_to(blob)
    restored = CheckpointLayers()
    restored.conv("test_conv", 2, 3, 3)
    restored.norm("test_bn", 3)
    load_safetensors(restored, native)
    for key, expected in tensors.items():
        layer, attribute = key.split(".")
        np.testing.assert_array_equal(
            np.asarray(
                cast(NativeArray, getattr(restored.layer(layer), attribute)), dtype=np.float32
            ),
            np.asarray(expected, dtype=np.float32),
        )


@pytest.mark.parametrize("invalid", ["shape", "dtype", "nonfinite"])
def test_native_weights_reject_invalid_tensors(tmp_path: Path, invalid: str) -> None:
    """Reject checkpoint tensors with wrong shapes, dtypes, or non-finite values."""
    model = CheckpointLayers()
    model.conv("test_conv", 1, 2, 3)
    weight = mx.ones((2, 3, 3, 3, 1), dtype=mx.float32)
    if invalid == "shape":
        weight = weight[:1]
    elif invalid == "dtype":
        weight = weight.astype(mx.float16)
    else:
        weight = weight * float("nan")
    path = tmp_path / "invalid.safetensors"
    cast(NativeCheckpointWriter, mx).save_safetensors(
        str(path), {"test_conv.weight": weight, "test_conv.bias": mx.zeros((2,), dtype=mx.float32)}
    )
    with pytest.raises(ModelExecutionError):
        load_safetensors(model, path)
