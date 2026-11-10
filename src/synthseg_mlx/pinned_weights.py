"""Locate the SynthSeg checkpoints a variant needs."""

from __future__ import annotations

from pathlib import Path

from synthseg_mlx.variants import SynthSegVariant


def checkpoint_paths(
    models_dir: Path, variant: SynthSegVariant, *, parc: bool = True
) -> dict[str, Path]:
    """Locate enabled safetensors files; model loading validates their contents."""
    paths = {
        name: models_dir / name for name in variant.checkpoints
        if parc or name != "synthseg_parc_2.0.safetensors"
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    return paths
