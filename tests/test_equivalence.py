"""Integer QC cropping against upstream TensorFlow."""

from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from recording_types import Recording

from synthseg_mlx import crop_for_qc
from synthseg_mlx._typing import IndexArray


@pytest.mark.parametrize("target", [8, 32])
@pytest.mark.parametrize("index", [0, 1, 2], ids=["centered", "edge", "excluded"])
def test_qc_crop_matches_tensorflow(
    reference_fixtures: dict[str, Recording],
    target: int,
    index: int,
) -> None:
    """Compare native QC cropping with recorded TensorFlow outputs."""
    fixture = reference_fixtures[f"qc_crop_{target}"]
    actual = cast(
        IndexArray, np.array(crop_for_qc(mx.array(fixture["input"][index]), target=target))
    )
    np.testing.assert_array_equal(actual, fixture["expected"][index])
    assert actual.dtype == np.int32
