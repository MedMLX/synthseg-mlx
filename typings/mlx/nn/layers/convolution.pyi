"""Consumed Conv3d API from MLX 0.32.3; no convolution implementation."""

from mlx.core import array
from mlx.nn.layers.base import Module

class Conv3d(Module):
    weight: array
    # Present when constructed with bias=True, as every SynthSeg layer is.
    bias: array
    padding: int | tuple[int, int, int]
    stride: int | tuple[int, int, int]
    dilation: int | tuple[int, int, int]
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ) -> None:
        """Construct a three-dimensional native convolution layer."""
        ...
    def __call__(self, x: array) -> array:
        """Apply the convolution to a channels-last native array."""
        ...
