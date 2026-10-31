"""Stage-by-stage evaluation bounds SynthSeg's peak MLX memory."""

from collections.abc import Callable
from typing import cast

import mlx.core as mx
import numpy as np

from synthseg_mlx import SynthSegModel
from synthseg_mlx._typing import FloatArray, NativeArrayApi

Outputs = tuple[mx.array, mx.array, mx.array]


def _peak(run: Callable[[], Outputs]) -> tuple[int, int, list[FloatArray]]:
    """Return peak bytes above what was held, bytes cached once evaluated, and the outputs."""

    native = cast(NativeArrayApi, mx)
    mx.clear_cache()
    mx.reset_peak_memory()
    held = mx.get_active_memory()
    outputs = run()
    native.eval(*outputs)
    cached = mx.get_cache_memory()
    copies = [cast(FloatArray, np.array(item)) for item in outputs]
    return mx.get_peak_memory() - held, cached, copies


def test_predict_bounds_peak_memory_with_identical_outputs() -> None:
    # One evaluation of the whole graph keeps every layer's temporaries alive;
    # measured at 48^3 the staged peak is about a third of it (a fifth at 64^3).
    """Compare bounded and unbounded prediction outputs and peak memory."""
    labels = np.array([0, 3, 7, 42], dtype=np.int32)
    indices = np.arange(4, dtype=np.int32)
    model = SynthSegModel(labels, labels, indices, indices, qc_shape=48)
    image = mx.array(
        np.random.default_rng(20261016).normal(size=(1, 48, 48, 48, 1)).astype(np.float32)
    )
    cast(NativeArrayApi, mx).eval(image, model.parameters())

    whole, _, expected = _peak(lambda: model(image))
    caller_limit = 3 * 2**30
    original_limit = mx.set_cache_limit(caller_limit)
    try:
        staged, cached, actual = _peak(lambda: model.predict(image))
        # Freed buffers were not cached during prediction, and the limit is the caller's again.
        assert cached == 0
        assert mx.set_cache_limit(caller_limit) == caller_limit
    finally:
        mx.set_cache_limit(original_limit)

    for field, staged_output, whole_output in zip(
        ("segmentation", "parcellation", "qc"), actual, expected, strict=True
    ):
        np.testing.assert_array_equal(staged_output, whole_output, err_msg=field)
    assert staged < 0.5 * whole, f"staged peak {staged} bytes, whole-graph peak {whole} bytes"
