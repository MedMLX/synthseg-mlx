"""Canonical SynthSeg release and processing choices, following the upstream CLI."""

from enum import StrEnum
from typing import Literal, cast

from medmlx_core.errors import InvalidInputError

type ArithmeticProfile = Literal["source-fp32", "compensated-fp32"]

# BBillot/SynthSeg revision whose prediction graphs and CLI rules the variants follow.
UPSTREAM_REVISION = "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d"


class SynthSegVariant(StrEnum):
    """Released SynthSeg configurations; each value is the variant's short name."""

    STANDARD = "standard"
    FAST = "fast"
    ROBUST = "robust"
    V1 = "v1"
    V1_FAST = "v1_fast"

    @property
    def robust(self) -> bool:
        """Return whether this variant uses the robust 2.0 cascade."""
        return self is self.ROBUST

    @property
    def fast(self) -> bool:
        """True when left/right averaging and per-class cleanup are skipped (robust implies it)."""
        return self in (self.FAST, self.ROBUST, self.V1_FAST)

    @property
    def v1(self) -> bool:
        """Return whether this variant uses the version 1 source assets."""
        return self in (self.V1, self.V1_FAST)

    @property
    def segmentation_checkpoint(self) -> str:
        """Safetensors file for the selected segmentation network."""
        if self.robust:
            return "synthseg_robust_2.0.safetensors"
        return "synthseg_1.0.safetensors" if self.v1 else "synthseg_2.0.safetensors"

    @property
    def checkpoints(self) -> tuple[str, ...]:
        """Safetensors files required by this variant, including 2.0 parcellation and QC."""
        return (
            self.segmentation_checkpoint,
            "synthseg_parc_2.0.safetensors",
            "synthseg_qc_2.0.safetensors",
        )

    @property
    def neutral_labels(self) -> int:
        """Count of left/right-neutral labels at the start of the segmentation label list."""
        return 18 if self.v1 else 19


def parse_variant(*, robust: bool = False, fast: bool = False, v1: bool = False) -> SynthSegVariant:
    """Map the CLI-style flags to a variant.

    Raises:
        InvalidInputError: If a flag is not a bool, or ``robust`` is combined with ``v1``.
    """
    for name, value in (("robust", robust), ("fast", fast), ("v1", v1)):
        if not isinstance(cast(object, value), bool):
            raise InvalidInputError(f"SynthSeg {name} must be true or false")
    if robust:
        if v1:
            raise InvalidInputError("SynthSeg v1 cannot be used with robust (robust requires 2.0)")
        return SynthSegVariant.ROBUST
    if v1:
        return SynthSegVariant.V1_FAST if fast else SynthSegVariant.V1
    return SynthSegVariant.FAST if fast else SynthSegVariant.STANDARD
