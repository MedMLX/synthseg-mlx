"""Offline reuse of weights prepared through the shared download contract."""

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from medmlx_core import weights as shared_weights

from synthseg_mlx import medmlx


class WeightCheckReached(Exception):
    """Stop at the weight file check before loading checkpoints."""


def test_preloaded_weights_are_reused_with_the_hub_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-download by registered ID, then reach the same model folder without network."""
    offline = False

    def model_info(_repo_id: str) -> SimpleNamespace:
        assert not offline, "Pre-downloaded weights must be reusable with the Hub offline"
        return SimpleNamespace(sha="a" * 40)

    def snapshot_download(_repo_id: str, *, local_dir: Path, **_options: object) -> str:
        assert not offline, "An offline run must not download another model folder"
        local_dir.mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=model_info),
        snapshot_download=snapshot_download,
    )

    def import_hub(_name: str) -> SimpleNamespace:
        return hub

    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(shared_weights, "import_module", import_hub)
    preloaded = shared_weights.weights_dir(str(medmlx.MODEL["id"]))
    assert shared_weights.weights_revision(preloaded) == "a" * 40
    offline = True
    selected: list[Path] = []

    def check_weights(folder: Path, _variant: medmlx.SynthSegVariant) -> NoReturn:
        selected.append(folder)
        raise WeightCheckReached("weight check reached")

    monkeypatch.setattr(medmlx, "_require_weight_files", check_weights)
    image = tmp_path / "brain.nii.gz"
    image.write_bytes(b"not read before the weight check")
    with pytest.raises(WeightCheckReached, match="weight check reached"):
        medmlx.run(image_path=image, output_dir=tmp_path / "out")
    assert selected == [preloaded]
