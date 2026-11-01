"""Session fixtures shared by the tests: the recorded upstream processing arrays."""

from pathlib import Path

import pytest
from recording_types import Recording
from variant_fixtures import load_recording

FIXTURES = Path(__file__).parent / "fixtures" / "processing"


@pytest.fixture(scope="session")
def reference_fixtures() -> dict[str, Recording]:
    """Load available recorded processing fixtures by name."""
    return {path.stem: load_recording(path) for path in FIXTURES.glob("*.npz")}
