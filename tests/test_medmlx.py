"""Runner argument rejection before any weight download, without Metal."""

from pathlib import Path
from typing import NotRequired, TypedDict, cast

import nibabel as nib_runtime
import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from synthseg_mlx import medmlx
from synthseg_mlx._typing import NiftiApi
from synthseg_mlx.variants import ArithmeticProfile

nib = cast(NiftiApi, nib_runtime)


class RunArguments(TypedDict):
    image_path: Path
    output_dir: Path
    ct: NotRequired[bool]
    arithmetic_profile: NotRequired[ArithmeticProfile]


@pytest.mark.parametrize(
    "problem", ["bad-suffix", "out-of-range", "non-bool", "existing-output"]
)
def test_run_rejects_arguments_before_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Each category of bad argument fails before the weights are requested."""
    import medmlx_core.weights as managed_weights

    def download(_model_id: str) -> Path:
        """Fail if the runner asks for weights before checking its arguments."""
        pytest.fail("weights were requested before the arguments were checked")

    monkeypatch.setattr(managed_weights, "weights_dir", download)
    image = tmp_path / "image.nii.gz"
    data = np.random.default_rng(5).random((12, 13, 14)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(image))
    output = tmp_path / "out"
    arguments: dict[str, object] = {"image_path": image, "output_dir": output}
    if problem == "bad-suffix":
        arguments["image_path"] = tmp_path / "image.mgz"
        (tmp_path / "image.mgz").write_bytes(b"")
    elif problem == "out-of-range":
        arguments["arithmetic_profile"] = "fp16"
    elif problem == "non-bool":
        arguments["ct"] = "yes"
    else:
        output.mkdir()
        (output / "mask.nii.gz").write_bytes(b"previous")
    with pytest.raises(InvalidInputError):
        medmlx.run(**cast(RunArguments, arguments))


def test_declared_labels_match_the_bundled_label_map() -> None:
    """The discovery labels equal the 2.0 map, which also covers every v1 label."""
    names = medmlx.label_names()
    assert medmlx.MODEL["labels"] == {str(value): name for value, name in names.items()}
    assert medmlx.label_names(v1=True).items() <= names.items()
