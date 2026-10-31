"""Decimal reporting precision is separate from binary framework arithmetic."""

import numpy as np
import pytest
from variant_fixtures import assert_reporting_units

from synthseg_mlx import qc_scores


@pytest.mark.parametrize(("reference", "decimals"), [(2291.366, 3), (0.1234, 4)])
def test_reporting_bound_accepts_one_unit_and_rejects_two(reference: float, decimals: int) -> None:
    """Check reporting comparisons at the configured decimal-unit boundary."""
    unit = 10.0**-decimals
    assert_reporting_units([reference + unit], [reference], decimals=decimals)
    with pytest.raises(AssertionError, match="Reporting-unit differences"):
        assert_reporting_units([reference + 2 * unit], [reference], decimals=decimals)
    with pytest.raises(AssertionError):
        assert_reporting_units([np.nan], [reference], decimals=decimals)


@pytest.mark.parametrize("options", [{}, {"fast": True}, {"robust": True}, {"v1": True}])
def test_quality_scores_drop_background_clip_and_round(options: dict[str, bool]) -> None:
    """Check quality scores omit background and apply clipping and rounding."""
    raw = np.array([[0.8, -0.02, 0.123456, 1.25]], dtype=np.float32)
    actual = qc_scores(raw, **options)
    assert actual.dtype == (np.float64 if not options else np.float32)
    assert_reporting_units(actual, [0, 0.1235, 1], decimals=4)
