"""Consumed MLX 0.32.3 Module API; actual runtime class remains MLX's.

Container methods are explicit because native update returns self, unlike
dict.update. This partial declaration does not change native dict ancestry.
"""

from collections.abc import Iterator
from typing import Self

from mlx.core import array

type ParameterValue = array | dict[str, ParameterValue] | list[ParameterValue]

class Module:
    def __init__(self) -> None:
        """Initialize the native module container."""
        ...
    def __contains__(self, key: object) -> bool:
        """Return whether a named member exists."""
        ...
    def __getitem__(self, key: str) -> object:
        """Return a named module member."""
        ...
    def __setitem__(self, key: str, value: object) -> None:
        """Set a named module member."""
        ...
    def __delitem__(self, key: str) -> None:
        """Remove a named module member."""
        ...
    def __iter__(self) -> Iterator[str]:
        """Iterate member names."""
        ...
    def __len__(self) -> int:
        """Return the number of named members."""
        ...
    def __getattr__(self, key: str) -> object:
        """Return a member accessed as an attribute."""
        ...
    def __setattr__(self, key: str, val: object) -> None:
        """Assign a member as an attribute."""
        ...
    def parameters(self) -> dict[str, ParameterValue]:
        """Return the nested native parameter tree."""
        ...
    def load_weights(
        self, file_or_weights: str | list[tuple[str, array]], strict: bool = True
    ) -> Self:
        """Load named weights and enforce strict matching when requested."""
        ...
    def save_weights(self, file: str) -> None:
        """Write module weights to a file."""
        ...
    def update(self, parameters: dict[str, ParameterValue], strict: bool = True) -> Self:
        """Update the native parameter tree and return this module."""
        ...
    def eval(self) -> Self:
        """Set inference mode and return this module."""
        ...
