"""Repository-used MLX 0.32.4 ``mlx/nn/layers/convolution.py`` APIs, from the installed source."""

from mlx.core import array
from mlx.nn.layers.base import Module

type Shape = int | tuple[int, ...]

class Conv3d(Module):
    weight: array
    bias: array
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...
