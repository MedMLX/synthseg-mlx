"""Native float32 MLX implementation of standard, robust, fast and v1 graphs.

MLX ``Conv3d`` consumes NDHWC tensors and safetensors weights in
``(C_out, D, H, W, C_in)`` order.
"""

from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from medmlx_core.compensated import CompensatedFloat32Operators
    from mlx.core import array as NativeArray
    from mlx.nn.layers.base import Module as NativeModule
    from mlx.nn.layers.convolution import Conv3d as NativeConv3d

import numpy as np
from medmlx_core.errors import ModelExecutionError
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime

from synthseg_mlx._typing import (
    IndexArray,
    MetalKernel,
    NativeArrayApi,
    NativeCheckpointApi,
    NativeOperation,
    NeuralRuntime,
)
from synthseg_mlx.variants import ArithmeticProfile, SynthSegVariant, parse_variant


def _native_arrays():
    """Admit the MLX runtime and return its core array module."""
    import_mlx()
    import mlx.core as mx

    return mx


mx = _native_arrays()

nn = cast(NeuralRuntime, import_module("mlx.nn"))


def _evaluate(value: NativeArray) -> NativeArray:
    """Evaluate ``value`` now, so later graph building starts from stored data."""

    cast(NativeArrayApi, mx).eval(value)
    return value


def _defer(value: NativeArray) -> NativeArray:
    """Leave ``value`` lazy; the unbounded counterpart of ``_evaluate``."""

    return value


def load_safetensors(module: NativeModule, path: str | Path) -> None:
    """Load native parameters and require finite float32 values and matching shapes."""
    api = cast(NativeCheckpointApi, mx)
    try:
        # An explicit format also supports Hub-cache symlinks to extensionless blobs.
        weights = api.load(str(path), format="safetensors")
    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelExecutionError(f"Cannot read SynthSeg safetensors {path}: {exc}") from exc
    if any(value.dtype != api.float32 for value in weights.values()):
        raise ModelExecutionError(f"SynthSeg safetensors {path} must contain float32 tensors")
    finite = [api.all(api.isfinite(value)) for value in weights.values()]
    api.eval(*finite)
    if not all(bool(value.item()) for value in finite):
        raise ModelExecutionError(f"SynthSeg safetensors {path} contains non-finite values")
    try:
        module.load_weights(list(weights.items()), strict=True)
    except ValueError as exc:
        raise ModelExecutionError(f"Invalid SynthSeg safetensors {path}: {exc}") from exc


class CompensatedConv3d(nn.Conv3d):
    """MLX-layout parameters with compensated Metal channel/spatial reduction."""

    def __init__(
        self, inputs: int, outputs: int, kernel: int, operators: CompensatedFloat32Operators
    ) -> None:
        """Initialize checkpoint-shaped convolution and compensated operators."""
        super().__init__(inputs, outputs, kernel, padding=kernel // 2)
        self._accuracy_ops = operators
        self._accuracy_padding = kernel // 2

    def __call__(self, values: NativeArray) -> NativeArray:
        """Apply compensated convolution while preserving the native channel layout."""
        result = self._accuracy_ops.conv(
            mx.transpose(values, (0, 4, 1, 2, 3)),
            mx.transpose(self.weight, (0, 4, 1, 2, 3)),
            self.bias,
            padding=self._accuracy_padding,
            stride=1,
        )
        return mx.transpose(result, (0, 2, 3, 4, 1))


class CheckpointLayers(nn.Module):
    """MLX layers whose names match the source graph."""

    def __init__(self, *, accuracy_ops: CompensatedFloat32Operators | None = None) -> None:
        """Initialize named checkpoint layers and optional arithmetic operators."""
        super().__init__()
        self._accuracy_ops = accuracy_ops
        self.layer_names: list[str] = []

    def conv(self, name: str, inputs: int, outputs: int, kernel: int) -> None:
        """Add a named native or compensated three-dimensional convolution."""
        layer = (
            nn.Conv3d(inputs, outputs, kernel, padding=kernel // 2)
            if self._accuracy_ops is None
            else CompensatedConv3d(inputs, outputs, kernel, self._accuracy_ops)
        )
        setattr(self, name, layer)
        self.layer_names.append(name)

    def norm(self, name: str, channels: int) -> None:
        """Add a named inference batch-normalization layer."""
        setattr(self, name, BatchNorm3d(channels, eps=1e-3, accuracy_ops=self._accuracy_ops))
        self.layer_names.append(name)

    def layer(self, name: str) -> NativeOperation:
        """Return a previously registered named checkpoint layer."""
        return cast(NativeOperation, getattr(self, name))

class BatchNorm3d(nn.Module):
    """Inference-only channels-last batch normalization for five-dimensional tensors.

    MLX's generic ``nn.BatchNorm`` intentionally accepts image/video layouts up
    to four dimensions.  SynthSeg uses NDHWC volumes, so this directly applies
    the frozen moving statistics over the final channel axis.
    """

    def __init__(
        self,
        channels: int,
        *,
        eps: float,
        accuracy_ops: CompensatedFloat32Operators | None = None,
    ) -> None:
        """Initialize frozen per-channel statistics and optional compensated arithmetic."""
        super().__init__()
        self._accuracy_ops = accuracy_ops
        self.eps = eps
        self.weight = mx.ones((channels,), dtype=mx.float32)
        self.bias = mx.zeros((channels,), dtype=mx.float32)
        self.running_mean = mx.zeros((channels,), dtype=mx.float32)
        self.running_var = mx.ones((channels,), dtype=mx.float32)

    def __call__(self, x: NativeArray) -> NativeArray:
        """Normalize a five-dimensional channels-last tensor using stored statistics."""
        if x.ndim != 5 or x.shape[-1] != self.weight.shape[0]:
            raise ModelExecutionError("SynthSeg MLX batch normalization expects NDHWC tensors")
        if self._accuracy_ops is not None:
            return self._accuracy_ops.batch_norm_ndhwc(
                x, self.running_mean, self.running_var, self.weight, self.bias
            )
        return (x - self.running_mean) * mx.rsqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


def repeat_upsample(x: NativeArray, skip: NativeArray | None) -> NativeArray:
    """Source-equivalent 2x nearest-neighbor upsampling before skip concatenation."""

    batch, depth, height, width, channels = x.shape
    repeated = mx.reshape(
        mx.broadcast_to(
            mx.reshape(x, (batch, depth, 1, height, 1, width, 1, channels)),
            (batch, depth, 2, height, 2, width, 2, channels),
        ),
        (batch, 2 * depth, 2 * height, 2 * width, channels),
    )
    if skip is None:
        return repeated
    if repeated.shape[:4] != skip.shape[:4]:
        raise ModelExecutionError(
            "SynthSeg MLX cannot reproduce a source decoder resize for this input grid"
        )
    return mx.concatenate([skip, repeated], axis=-1)


class UNet(CheckpointLayers):
    """The source five-level, ELU, batch-normalized SynthSeg U-Net.

    Input and output are channels-last: ``(B, D, H, W, channels)`` in, softmax
    probabilities ``(B, D, H, W, labels)`` out. Four 2x poolings mean D, H and W must be
    divisible by 16. Layer names are ``<prefix>_...`` and match the source graph
    (``unet`` segmentation, ``unet_parc`` parcellation, ``l2l`` denoiser, ``unet2``
    robust second stage). ``features`` is the first level's width and doubles per level;
    ``skip_n_concatenations`` omits the skip connection of that many final decoder
    levels (2 for the robust denoiser).
    """

    def __init__(
        self,
        channels: int,
        labels: int,
        prefix: str,
        *,
        features: int = 24,
        kernel_size: int = 3,
        skip_n_concatenations: int = 0,
        accuracy_ops: CompensatedFloat32Operators | None = None,
    ) -> None:
        """Construct the source five-level encoder and decoder for the requested labels."""
        super().__init__(accuracy_ops=accuracy_ops)
        if features < 1 or kernel_size not in (3, 5) or skip_n_concatenations not in range(5):
            raise ModelExecutionError("Invalid SynthSeg U-Net architecture")
        self.prefix = prefix
        self.skip_n_concatenations = skip_n_concatenations
        for level in range(5):
            width = features * 2**level
            self.conv(f"{prefix}_conv_downarm_{level}_0", channels, width, kernel_size)
            self.conv(f"{prefix}_conv_downarm_{level}_1", width, width, kernel_size)
            self.norm(f"{prefix}_bn_down_{level}", width)
            channels = width
        for level in range(4):
            width = features * 2 ** (3 - level)
            incoming = channels + width if level < 4 - skip_n_concatenations else channels
            self.conv(f"{prefix}_conv_uparm_{level + 5}_0", incoming, width, kernel_size)
            self.conv(f"{prefix}_conv_uparm_{level + 5}_1", width, width, kernel_size)
            self.norm(f"{prefix}_bn_up_{level}", width)
            channels = width
        self.conv(f"{prefix}_likelihood", channels, labels, 1)
        self.pool = nn.MaxPool3d(2, stride=2)

    def __call__(self, x: NativeArray, *, bounded: bool = False) -> NativeArray:
        """Return per-voxel label probabilities; see ``logits`` for ``bounded``."""
        logits = self.logits(x, bounded=bounded)
        return (
            mx.softmax(logits, axis=-1)
            if self._accuracy_ops is None
            else self._accuracy_ops.softmax_last(logits)
        )

    def _elu(self, values: NativeArray) -> NativeArray:
        """Apply the selected standard or compensated ELU operation."""
        return nn.elu(values) if self._accuracy_ops is None else self._accuracy_ops.elu(values)

    def logits(self, x: NativeArray, *, bounded: bool = False) -> NativeArray:
        """Return the released likelihood layer before the probability softmax.

        ``bounded`` evaluates every level before building the next, and each
        full-resolution operation on its own; see ``SynthSegModel.predict``.
        """
        settle = _evaluate if bounded else _defer
        skips: list[NativeArray] = []
        for level in range(5):
            full_resolution = settle if level == 0 else _defer
            x = full_resolution(self._elu(self.layer(f"{self.prefix}_conv_downarm_{level}_0")(x)))
            x = settle(self._elu(self.layer(f"{self.prefix}_conv_downarm_{level}_1")(x)))
            # The source stores skip tensors before batch normalization.
            if level < 4:
                skips.append(x)
            x = self.layer(f"{self.prefix}_bn_down_{level}")(x)
            if level < 4:
                x = settle(self.pool(x))
        for level in range(4):
            full_resolution = settle if level == 3 else _defer
            skip = skips.pop()
            x = full_resolution(
                repeat_upsample(x, skip if level < 4 - self.skip_n_concatenations else None)
            )
            x = full_resolution(self._elu(self.layer(f"{self.prefix}_conv_uparm_{level + 5}_0")(x)))
            x = self._elu(self.layer(f"{self.prefix}_conv_uparm_{level + 5}_1")(x))
            x = settle(self.layer(f"{self.prefix}_bn_up_{level}")(x))
        return settle(self.layer(f"{self.prefix}_likelihood")(x))


class RobustSegmentation(nn.Module):
    """Upstream S1 -> hard groups -> D -> hard groups plus image -> S2 cascade.

    S1 and the denoiser D work on ``groups`` coarse tissue classes; each transition is an
    argmax followed by one-hot encoding. The output is the final U-Net's probabilities
    ``(B, D, H, W, labels)``.
    """

    def __init__(
        self,
        groups: int,
        labels: int,
        *,
        features: int = 24,
        denoiser_features: int = 16,
    ) -> None:
        """Construct the coarse, denoiser, and fine segmentation networks."""
        super().__init__()
        self.groups = groups
        self.coarse = UNet(1, groups, "unet", features=features)
        self.denoiser = UNet(
            groups,
            groups,
            "l2l",
            features=denoiser_features,
            kernel_size=5,
            skip_n_concatenations=2,
        )
        channels = 1 + (groups - 1 if groups <= 2 else groups)
        self.fine = UNet(channels, labels, "unet2", features=features)

    def hard_groups(self, probabilities: NativeArray) -> NativeArray:
        """Argmax over the last axis, one-hot encoded back to ``groups`` float32 channels."""

        indices = mx.argmax(probabilities, axis=-1)
        return (indices[..., None] == mx.arange(self.groups)).astype(mx.float32)

    def __call__(self, image: NativeArray, *, bounded: bool = False) -> NativeArray:
        """Run the cascade on an image ``(B, D, H, W, 1)`` and return label probabilities."""
        groups = self.hard_groups(self.coarse(image, bounded=bounded))
        groups = self.hard_groups(self.denoiser(groups, bounded=bounded))
        if self.groups <= 2:
            groups = groups[..., 1:]
        return self.fine(mx.concatenate([image, groups], axis=-1), bounded=bounded)

# QC has one active input class per voxel. Spatial/class/output weight order
# makes neighboring output channels read neighboring coefficients on Metal.
_qc_class_convolution = cast(
    MetalKernel,
    mx.fast.metal_kernel(
        name="synthseg_qc_class_convolution",
        input_names=["indices", "weights", "bias"],
        output_names=["out"],
        source="""
uint wc=thread_position_in_grid.x;
if (wc>=W*O) return;
uint row=thread_position_in_grid.y;
uint batch=thread_position_in_grid.z;
int d=row/H,h=row%H,w=wc/O;
uint co=wc%O;
ulong base=ulong(batch)*D*H*W;
float value=0.f;
#pragma unroll
for (int kd=0;kd<5;kd++) {
    int zd=d+kd-2;
    if (zd<0||zd>=D) continue;
    #pragma unroll
    for (int kh=0;kh<5;kh++) {
        int yh=h+kh-2;
        if (yh<0||yh>=H) continue;
        #pragma unroll
        for (int kw=0;kw<5;kw++) {
            int xw=w+kw-2;
            if (xw<0||xw>=W) continue;
            int label=indices[base+(ulong(zd)*H+yh)*W+xw];
            if(label>=0&&label<C) value=fma(1.f,weights[(((kd*5+kh)*5+kw)*C+label)*O+co],value);
        }
    }
}
out[(base+ulong(row)*W+w)*O+co]=value+bias[co];
""",
        compile_options={"math_mode": "safe"},
    ),
)

# Keep each FP32 normalization rounding step; volatile products prevent FMA
# contraction. Evaluate all eight source pool values, including negative gamma.
_qc_residual_pool = cast(
    MetalKernel,
    mx.fast.metal_kernel(
        name="synthseg_qc_residual_norm_pool",
        input_names=["x", "residual", "mean", "scale", "weight", "bias"],
        output_names=["out"],
        source="""
uint wc=thread_position_in_grid.x;
if (wc>=(W/2)*C) return;
uint row=thread_position_in_grid.y;
uint batch=thread_position_in_grid.z;
int d=row/(H/2),h=row%(H/2),w=wc/C;
uint channel=wc%C;
ulong base=ulong(batch)*D*H*W*C;
float pooled=-INFINITY;
#pragma unroll
for(int kd=0;kd<2;kd++) {
    #pragma unroll
    for(int kh=0;kh<2;kh++) {
        #pragma unroll
        for(int kw=0;kw<2;kw++) {
            ulong offset=base+((ulong(2*d+kd)*H+2*h+kh)*W+2*w+kw)*C+channel;
            float value=x[offset]+residual[offset];
            value=isnan(value)?value:max(value,0.f);
            volatile float scaled=(value-mean[channel])*scale[channel];
            volatile float weighted=scaled*weight[channel];
            value=weighted+bias[channel];
            pooled=(isnan(pooled)||isnan(value))?NAN:max(pooled,value);
        }
    }
}
ulong output=(((ulong(batch)*(D/2)+d)*(H/2)+h)*(W/2)+w)*C+channel;
out[output]=pooled;
""",
        compile_options={"math_mode": "safe"},
    ),
)


class QCNet(CheckpointLayers):
    """The source residual QC encoder and global regional-score head.

    Input is a one-hot QC class map ``(B, D, H, W, labels)`` (or, via ``score_indices``,
    its int32 class indices); output is one non-negative score per class ``(B, labels)``,
    the spatial mean of the final activations. ``qc_scores`` later drops the background
    entry and clips to [0, 1].
    """

    def __init__(self, labels: int, *, features: int = 24) -> None:
        """Construct the residual quality-control encoder and scoring layers."""
        super().__init__()
        channels = labels
        for level in range(4):
            width = features * 2**level
            self.conv(f"qc_conv_downarm_{level}_0", channels, width, 5)
            self.conv(f"qc_conv_downarm_{level}_1", width, width, 5)
            self.conv(f"qc_expand_down_merge_{level}", channels, width, 5)
            self.norm(f"qc_bn_down_{level}", width)
            channels = width
        self.conv("qc_final_conv_0", channels, labels, 5)
        self.conv("qc_final_conv_1", labels, labels, 5)
        self.pool = nn.MaxPool3d(2, stride=2)

    def _norm_pool(self, x: NativeArray, residual: NativeArray, level: int) -> NativeArray:
        # ReLU(x + residual) -> frozen batch norm -> 2x2x2 max pool, fused on Metal when
        # the operands are same-shaped float32 NDHWC tensors.
        """Apply residual addition, activation, normalization, and pooling."""
        norm = cast(BatchNorm3d, self.layer(f"qc_bn_down_{level}"))
        stats = (norm.running_mean, norm.running_var, norm.weight, norm.bias)
        channels = x.shape[-1]
        # The fused kernel requires matching FP32 volumes and channel vectors.
        # Retain native broadcasting and empty-pool behavior for other layer states.
        if (
            x.shape != residual.shape
            or min(x.shape[1:4]) < 2
            or x.dtype != mx.float32
            or residual.dtype != mx.float32
            or any(value.shape != (channels,) or value.dtype != mx.float32 for value in stats)
        ):
            return self.pool(norm(nn.relu(x + residual)))
        batch, depth, height, width, _ = x.shape
        return _qc_residual_pool(
            inputs=[
                x,
                residual,
                norm.running_mean,
                mx.rsqrt(norm.running_var + norm.eps),
                norm.weight,
                norm.bias,
            ],
            template=list(dict(D=depth, H=height, W=width, C=channels).items()),
            grid=(
                (((width // 2 * channels + 255) // 256) * 256),
                depth // 2 * (height // 2),
                batch,
            ),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, depth // 2, height // 2, width // 2, channels)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )[0]

    def _finish_block(self, x: NativeArray, residual: NativeArray, level: int) -> NativeArray:
        # Second convolution of a QC level, then the residual add, norm and pool.
        """Finish a QC block with its second convolution and residual pooling."""
        x = self.layer(f"qc_conv_downarm_{level}_1")(x)
        return self._norm_pool(x, residual, level)

    def _forward(self, x: NativeArray, start: int) -> NativeArray:
        """Run remaining QC blocks and average regional scores spatially."""
        for level in range(start, 4):
            residual = nn.relu(self.layer(f"qc_expand_down_merge_{level}")(x))
            x = nn.relu(self.layer(f"qc_conv_downarm_{level}_0")(x))
            x = self._finish_block(x, residual, level)
        x = nn.relu(self.layer("qc_final_conv_0")(x))
        x = nn.relu(self.layer("qc_final_conv_1")(x))
        return x.mean(axis=(1, 2, 3))

    def __call__(self, x: NativeArray) -> NativeArray:
        """Run quality scoring on a channels-last one-hot class map."""
        return self._forward(x, 0)

    def _class_convolve(self, name: str, indices: NativeArray) -> NativeArray:
        # Same result as convolving the one-hot encoding of ``indices`` (NDHW int32) with
        # layer ``name``; indices outside [0, C) contribute nothing, as an all-zero one-hot.
        """Convolve integer class indices without materializing a one-hot volume."""
        layer = cast("NativeConv3d", self.layer(name))
        batch, depth, height, width = indices.shape
        channels, outputs = layer.weight.shape[-1], layer.weight.shape[0]
        return _qc_class_convolution(
            inputs=[indices, mx.transpose(layer.weight, (1, 2, 3, 4, 0)), layer.bias],
            template=list(dict(D=depth, H=height, W=width, C=channels, O=outputs).items()),
            grid=((((width * outputs + 255) // 256) * 256), depth * height, batch),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, depth, height, width, outputs)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )[0]

    def score_indices(self, indices: NativeArray) -> NativeArray:
        """Score an int32 NDHW class map without materializing its one-hot volume."""
        if indices.ndim != 4 or indices.dtype != mx.int32:
            raise ModelExecutionError("SynthSeg QC class indices require int32 NDHW tensors")
        channels = cast("NativeConv3d", self.layer("qc_conv_downarm_0_0")).weight.shape[-1]
        # Above 16 classes, conv3d visits channel blocks before spatial taps.
        # Preserve that accumulation order through the existing array interface.
        if channels > 16:
            one_hot = (indices[..., None] == mx.arange(channels, dtype=mx.int32)).astype(mx.float32)
            return self(one_hot)
        residual = nn.relu(self._class_convolve("qc_expand_down_merge_0", indices))
        x = nn.relu(self._class_convolve("qc_conv_downarm_0_0", indices))
        return self._forward(self._finish_block(x, residual, 0), 1)


def crop_for_qc(indices: NativeArray, target: int = 224) -> NativeArray:
    """Replicate the source QC crop while keeping label-map math in MLX.

    ``indices`` is one integer map ``(D, H, W)``. The bounding box of the voxels that
    are neither 0 nor 24 is centered in a ``target``-sided cube; the cube is padded
    with 0 wherever it extends past the volume. With no such voxels, the first
    ``min(size, target)`` voxels of each axis form the box. The value 24 is excluded
    exactly as in the recorded upstream ``MakeShape`` reference.

    Raises:
        ModelExecutionError: If ``indices`` is not three-dimensional.
    """

    if indices.ndim != 3:
        raise ModelExecutionError("SynthSeg QC expects one 3D segmentation-index map")
    mask = (indices != 0) & (indices != 24)
    # Keep the dynamically determined crop bounds as MLX scalars.  ``take``
    # with clamped coordinates and a native validity mask expresses every
    # centered crop (including one extending past an edge) without moving
    # bounds or label-map operations back to the host, or materializing a
    # padded full volume.
    result = indices
    for axis, size in enumerate(indices.shape):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = mx.any(mask, axis=other_axes)
        occupied_int = occupied.astype(mx.int32)
        has_region = mx.any(occupied)
        low = mx.where(
            has_region,
            mx.argmax(occupied_int).astype(mx.int32),
            mx.array(0, dtype=mx.int32),
        )
        high = mx.where(
            has_region,
            mx.array(size, dtype=mx.int32) - mx.argmax(mx.flip(occupied_int, axis=0)),
            mx.array(min(size, target), dtype=mx.int32),
        )
        delta = mx.array(target, dtype=mx.int32) - (high - low)
        low = low - mx.ceil(delta.astype(mx.float32) / 2).astype(mx.int32)
        coordinates = low + mx.arange(target, dtype=mx.int32)
        valid = (coordinates >= 0) & (coordinates < size)
        result = mx.take(result, mx.clip(coordinates, 0, size - 1), axis=axis)
        valid_shape = [1, 1, 1]
        valid_shape[axis] = target
        result = mx.where(mx.reshape(valid, valid_shape), result, 0)
    return result


def _gaussian_weight(sigma: float, *, accurate: bool = False) -> NativeArray:
    """Normalized 3-D Gaussian kernel ``(K, K, K)`` with ``K = 2 * (ceil(2.5 * sigma) // 2) + 1``.

    ``sigma`` is in voxels; 0 gives a 1x1x1 identity kernel. ``accurate`` builds the
    coefficients in float64 and rounds once to float32, for the compensated arithmetic
    profile.
    """
    width = int(math.ceil(2.5 * sigma) / 2) * 2 + 1
    if accurate:
        # Fixed coefficients use the source's double formula and one FP32 rounding.
        # Learned inference still executes entirely on Metal.
        coordinates = np.arange(width, dtype=np.float64) - (width - 1) / 2
        difference = np.stack(
            np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1
        )
        scale = sigma if sigma else 1.0
        norms = -(difference**2) / (2 * scale**2) - np.log(np.sqrt(2 * np.pi) * scale)
        values = np.exp(norms.sum(axis=-1))
        values /= values.sum()
        return mx.array(values.astype(np.float32))
    coordinates = mx.arange(width, dtype=mx.float32) - (width - 1) / 2
    scale = sigma if sigma else 1.0
    squared = coordinates**2
    log_kernel = -(squared[:, None, None] + squared[None, :, None] + squared[None, None, :]) / (
        2 * scale**2
    )
    kernel = mx.exp(log_kernel)
    return kernel / kernel.sum()


# Visit the same spatial taps as conv3d, omitting its zero cross-channel weights.
# Explicit fma preserves the float32 accumulation order without separable passes.
_gaussian_blur = cast(
    MetalKernel,
    mx.fast.metal_kernel(
        name="synthseg_channelwise_gaussian",
        input_names=["x", "weights"],
        output_names=["out"],
        source="""
uint wc = thread_position_in_grid.x;
if (wc >= W * C) return;
uint row = thread_position_in_grid.y;
uint batch = thread_position_in_grid.z;
int d = row / H, h = row % H, w = wc / C;
uint channel = wc % C;
ulong base = ulong(batch) * D * H * W * C;
float value = 0.0f;
#pragma unroll
for (int kd = 0; kd < K; ++kd) {
    int zd = d + kd - K / 2;
    if (zd < 0 || zd >= D) continue;
    #pragma unroll
    for (int kh = 0; kh < K; ++kh) {
        int yh = h + kh - K / 2;
        if (yh < 0 || yh >= H) continue;
        #pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            int xw = w + kw - K / 2;
            if (xw < 0 || xw >= W) continue;
            ulong offset = base + ((ulong(zd) * H + yh) * W + xw) * C + channel;
            value = fma(x[offset], weights[(kd * K + kh) * K + kw], value);
        }
    }
}
out[base + ulong(row) * W * C + wc] = value;
""",
        compile_options={"math_mode": "safe"},
    ),
)


class SynthSegModel(nn.Module):
    """Segmentation, cortical parcellation, and QC in one MLX-native graph.

    Takes a preprocessed float32 volume ``(B, D, H, W, 1)`` in RAS axis order and returns
    ``(segmentation, parcellation, qc)``: label posteriors ``(B, D, H, W, n_seg)``,
    parcellation posteriors ``(B, D, H, W, n_parc)`` and raw QC scores ``(B, n_qc)``.
    ``labels_segmentation`` and ``labels_parcellation`` are sorted unique label values.
    ``labels_qc`` maps each segmentation channel to a QC class index; its unique
    indices must be contiguous starting at zero. ``flip_indices`` (required unless
    fast) maps each segmentation channel to its left/right counterpart. ``sigma`` is
    the Gaussian smoothing (voxels) used in the segmentation path; the robust path
    returns its unsmoothed prediction. ``qc_shape`` is the QC crop size in voxels.
    """

    def __init__(
        self,
        labels_segmentation: IndexArray,
        labels_parcellation: IndexArray,
        labels_qc: IndexArray,
        flip_indices: IndexArray | None,
        *,
        sigma: float = 0.5,
        qc_shape: int = 224,
        robust: bool = False,
        fast: bool = False,
        v1: bool = False,
        labels_denoiser: IndexArray | None = None,
        features: int = 24,
        denoiser_features: int = 16,
        arithmetic_profile: ArithmeticProfile = "source-fp32",
    ) -> None:
        """Construct the selected segmentation, parcellation, and QC graphs."""
        super().__init__()
        if not np.array_equal(np.unique(labels_qc), np.arange(len(np.unique(labels_qc)))):
            raise ModelExecutionError("SynthSeg MLX requires contiguous QC class indices")
        self.variant = parse_variant(robust=robust, fast=fast, v1=v1)
        if arithmetic_profile not in {"source-fp32", "compensated-fp32"}:
            raise ModelExecutionError(f"Unknown SynthSeg arithmetic profile {arithmetic_profile!r}")
        if (
            arithmetic_profile == "compensated-fp32"
            and self.variant is not SynthSegVariant.STANDARD
        ):
            raise ModelExecutionError("Compensated SynthSeg supports the standard 2.0 graph")
        self.arithmetic_profile = arithmetic_profile
        self._accuracy_ops: CompensatedFloat32Operators | None = None
        if arithmetic_profile == "compensated-fp32":
            from medmlx_core.compensated import CompensatedFloat32Operators as AccuracyOperators

            self._accuracy_ops = AccuracyOperators(cast(MlxRuntime, mx), 1, 1e-3)
        if self.variant.robust:
            if labels_denoiser is None or not labels_denoiser.size:
                raise ModelExecutionError("SynthSeg robust requires explicit denoiser labels")
            self.segmentation = RobustSegmentation(
                len(np.unique(labels_denoiser)),
                len(labels_segmentation),
                features=features,
                denoiser_features=denoiser_features,
            )
        else:
            self.segmentation = UNet(
                1,
                len(labels_segmentation),
                "unet",
                features=features,
                accuracy_ops=self._accuracy_ops,
            )
        self.parcellation = UNet(
            3,
            len(labels_parcellation),
            "unet_parc",
            features=features,
            accuracy_ops=self._accuracy_ops,
        )
        self.qc = QCNet(len(np.unique(labels_qc)), features=features)
        if not self.variant.fast and flip_indices is None:
            raise ModelExecutionError("SynthSeg standard/v1 requires left-right flip indices")
        self.flip_indices = (
            None
            if self.variant.fast
            else mx.array(cast(IndexArray, flip_indices).astype(np.int32, copy=False))
        )
        self.cortex_labels = mx.array(
            ((labels_segmentation == 3) | (labels_segmentation == 42)).astype(np.bool_)
        )
        self.qc_labels = mx.array(labels_qc.astype(np.int32, copy=False))
        self.qc_classes = len(np.unique(labels_qc))
        self.sigma = sigma
        self.qc_shape = qc_shape
        self.segmentation_kernel = _gaussian_weight(sigma, accurate=self._accuracy_ops is not None)
        self.parcellation_kernel = _gaussian_weight(0.5, accurate=self._accuracy_ops is not None)
        self.eval()

    @classmethod
    def from_safetensors(
        cls,
        path_segmentation: str | Path,
        labels_segmentation: IndexArray,
        path_parcellation: str | Path,
        labels_parcellation: IndexArray,
        path_qc: str | Path,
        labels_qc: IndexArray,
        flip_indices: IndexArray | None,
        *,
        sigma: float = 0.5,
        qc_shape: int = 224,
        robust: bool = False,
        fast: bool = False,
        v1: bool = False,
        labels_denoiser: IndexArray | None = None,
        features: int = 24,
        denoiser_features: int = 16,
        arithmetic_profile: ArithmeticProfile = "source-fp32",
    ) -> SynthSegModel:
        """Construct the graph from three native MLX safetensors files."""

        model = cls(
            labels_segmentation,
            labels_parcellation,
            labels_qc,
            flip_indices,
            sigma=sigma,
            qc_shape=qc_shape,
            robust=robust,
            fast=fast,
            v1=v1,
            labels_denoiser=labels_denoiser,
            features=features,
            denoiser_features=denoiser_features,
            arithmetic_profile=arithmetic_profile,
        )
        model.load_safetensors(path_segmentation, path_parcellation, path_qc)
        return model

    def load_safetensors(
        self,
        path_segmentation: str | Path,
        path_parcellation: str | Path,
        path_qc: str | Path,
    ) -> None:
        """Load all three native MLX networks from finite float32 safetensors files."""
        load_safetensors(self.segmentation, path_segmentation)
        load_safetensors(self.parcellation, path_parcellation)
        load_safetensors(self.qc, path_qc)
        self.eval()
        cast(NativeArrayApi, mx).eval(self.parameters())

    def blur(self, x: NativeArray, *, parcellation: bool = False) -> NativeArray:
        """Smooth each channel of ``x`` ``(B, D, H, W, C)`` with the 3-D Gaussian kernel.

        Uses the segmentation kernel, or the fixed sigma-0.5 parcellation kernel when
        ``parcellation`` is set. Borders are zero-padded. Returns ``x`` unchanged when the
        segmentation sigma is 0.
        """
        if not parcellation and not self.sigma:
            return x
        kernel = self.parcellation_kernel if parcellation else self.segmentation_kernel
        if self._accuracy_ops is not None:
            return self._accuracy_ops.gaussian_ndhwc(x, kernel)
        batch, depth, height, width, channels = x.shape
        return _gaussian_blur(
            inputs=[x, kernel],
            template=list(dict(D=depth, H=height, W=width, C=channels, K=kernel.shape[0]).items()),
            grid=(((width * channels + 255) // 256) * 256, depth * height, batch),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )[0]

    def __call__(self, image: NativeArray) -> tuple[NativeArray, NativeArray, NativeArray]:
        """Build the lazy segmentation, parcellation and QC graph without evaluating it."""

        return self._forward(image, bounded=False)

    def predict(self, image: NativeArray) -> tuple[NativeArray, NativeArray, NativeArray]:
        """Evaluate the same graph one stage at a time and return evaluated outputs.

        Metal can keep an evaluation's temporary buffers until its command buffer
        completes, so evaluating the whole graph at once held every layer's
        temporaries together: over 30 GB on a 256x256x192 brain MR, against
        21 GB with one network per evaluation. Evaluating each U-Net level, and
        each full-resolution operation, before building the next keeps the peak
        near the largest single layer (the 72-channel full-resolution decoder
        convolution). The operations are unchanged, so results are bitwise
        identical to evaluating ``__call__``.

        MLX's free-buffer cache is disabled while predicting and the caller's
        limit is restored afterwards. Staged evaluation frees buffers of many
        sizes that are rarely reused, and the default limit (the memory limit)
        let the cache grow past the working set: on the same input the cache
        passed 24 GB before the run finished, while a small limit kept the
        live peak unchanged and held less cached memory.
        """

        previous = mx.set_cache_limit(0)
        try:
            outputs = self._forward(image, bounded=True)
            # Temporaries are freed as the GPU finishes; wait so none reach the cache.
            mx.synchronize()
            return outputs
        finally:
            mx.set_cache_limit(previous)

    def _forward(
        self, image: NativeArray, *, bounded: bool
    ) -> tuple[NativeArray, NativeArray, NativeArray]:
        """Compute segmentation, parcellation, and QC outputs for one native volume."""
        if image.ndim != 5 or image.shape[-1] != 1 or image.dtype != mx.float32:
            raise ModelExecutionError("SynthSeg MLX expects a float32 NDHWC single-channel volume")
        settle = _evaluate if bounded else _defer
        segmentation, cortex_prediction = self.segmentation_predictions(image, bounded=bounded)
        segmentation, cortex_prediction = settle(segmentation), settle(cortex_prediction)
        indices = mx.argmax(segmentation, axis=-1)

        cropped = mx.stack([crop_for_qc(item, self.qc_shape) for item in indices])
        qc_indices = self.qc_labels[cropped]
        qc = settle(self.qc.score_indices(qc_indices))

        cortex = self.cortex_labels[mx.argmax(cortex_prediction, axis=-1)]
        parcellation_input = mx.concatenate(
            [
                image,
                mx.logical_not(cortex)[..., None].astype(mx.float32),
                cortex[..., None].astype(mx.float32),
            ],
            axis=-1,
        )
        parcellation = settle(self.parcellation(parcellation_input, bounded=bounded))
        return segmentation, parcellation, qc

    def segmentation_predictions(
        self, image: NativeArray, *, bounded: bool = False
    ) -> tuple[NativeArray, NativeArray]:
        """Return ``(segmentation, cortex_prediction)`` posteriors.

        ``segmentation`` is what is reported and fed to QC; ``cortex_prediction`` is the
        copy whose argmax picks the cortex voxels that condition the parcellation network.
        They differ only for fast non-robust variants, where the cortex input is the
        smoothed prediction and the reported one is raw. Non-fast variants average the
        image's and its left-right flipped pass.
        """

        settle = _evaluate if bounded else _defer
        segmentation = self.segmentation(image, bounded=bounded)
        if self.variant.robust:
            return segmentation, segmentation
        if self.variant.fast:
            # Upstream retrieves unet_prediction for segmentation/QC, while the
            # cortex input to parcellation comes from the smoothed prediction.
            return segmentation, self.blur(segmentation)
        # The first pass is stored before the flipped pass is built.
        segmentation = settle(segmentation)
        # Flip augmentation: axis 1 of NDHWC is the first spatial axis, left-right after
        # RAS alignment. flip_indices swaps each label with its contralateral twin.
        flipped = self.segmentation(mx.flip(image, axis=1), bounded=bounded)
        if self.variant is SynthSegVariant.STANDARD and self._accuracy_ops is None:
            # Preserve the extracted default graph's arithmetic and bitwise fixtures.
            flipped = mx.flip(flipped, axis=1)[..., self.flip_indices]
            segmentation = self.blur((segmentation + flipped) * 0.5)
        else:
            # Compensated and v1 profiles retain upstream smoothing before the mean.
            segmentation = self.blur(segmentation)
            flipped = mx.flip(self.blur(flipped), axis=1)[..., self.flip_indices]
            segmentation = (segmentation + flipped) * 0.5
        return segmentation, segmentation


__all__ = [
    "CheckpointLayers",
    "QCNet",
    "SynthSegModel",
    "UNet",
    "crop_for_qc",
]
