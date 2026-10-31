"""Locate the SynthSeg checkpoints a variant needs."""

from __future__ import annotations

from pathlib import Path

from synthseg_mlx.variants import SynthSegVariant


def checkpoint_paths(models_dir: Path, variant: SynthSegVariant) -> dict[str, Path]:
    """Locate the variant's safetensors files; model loading validates their contents."""
    paths = {name: models_dir / name for name in variant.checkpoints}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    return paths
