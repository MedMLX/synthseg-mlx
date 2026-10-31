"""Gaussian boundary, batch and strided-input behavior against SciPy."""

import math
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from scipy.ndimage import convolve

from synthseg_mlx import SynthSegModel
from synthseg_mlx._typing import FloatArray


@pytest.mark.parametrize(
    ("sigma", "parcellation", "shape", "strided"),
    [
        (0.0, False, (2, 1, 3, 5, 4), False),
        (0.5, False, (2, 5, 7, 9, 4), False),
        (0.5, False, (2, 5, 7, 9, 4), True),
        (1.6, False, (2, 1, 3, 5, 4), True),
        (0.0, True, (2, 1, 3, 5, 4), True),
    ],
)
def test_smoothing_matches_scipy(
    sigma: float, parcellation: bool, shape: tuple[int, ...], strided: bool
) -> None:
    """Compare native smoothing against the SciPy reference."""
    labels = np.array([0, 3, 7, 42], dtype=np.int32)
    indices = np.arange(4, dtype=np.int32)
    model = SynthSegModel(labels, labels, indices, indices, sigma=sigma, features=1)
    values = np.random.default_rng(20261009).normal(size=shape).astype(np.float32)
    native = mx.array(values)
    if strided:
        native = mx.flip(native.swapaxes(1, 3), axis=2)
        values = np.flip(values.swapaxes(1, 3), axis=2)

    effective_sigma = 0.5 if parcellation else sigma
    expected = values.copy()
    if effective_sigma:
        width = int(math.ceil(2.5 * effective_sigma) / 2) * 2 + 1
        coordinates = np.arange(width, dtype=np.float32) - (width - 1) / 2
        squared = coordinates**2
        kernel = np.exp(
            -(squared[:, None, None] + squared[None, :, None] + squared[None, None, :])
            / (2 * effective_sigma**2)
        )
        kernel /= kernel.sum()
        for batch in range(values.shape[0]):
            for channel in range(values.shape[-1]):
                expected[batch, ..., channel] = convolve(
                    values[batch, ..., channel], kernel, mode="constant", cval=0
                )

    actual = cast(FloatArray, np.array(model.blur(native, parcellation=parcellation)))
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
