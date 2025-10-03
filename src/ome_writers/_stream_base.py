from __future__ import annotations

import abc
from abc import abstractmethod
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    import numpy as np

    from ._dimensions import Dimension


class OMEStream(abc.ABC):
    """Abstract base class for writing streams of image frames to OME-compliant files.

    This class defines the common interface for all OME stream writers, providing
    methods for creating streams, appending frames, flushing data, and managing
    stream lifecycle. Concrete implementations handle specific file formats
    (TIFF, Zarr) and storage backends.

    The class supports context manager protocol for automatic resource cleanup
    and provides path normalization utilities for cross-platform compatibility.
    """

    @abstractmethod
    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        """Create a new stream for path, dtype, and dimensions.

        Parameters
        ----------
        path : str
            Path to the output file or directory.
        dtype : np.dtype
            NumPy data type for the image data.
        dimensions : Sequence[Dimension]
            Sequence of dimension information describing the data structure.
        overwrite : bool, optional
            Whether to overwrite existing files or directories. Default is False.

        Returns
        -------
        Self
            The instance of the stream, allowing for chaining.
        """

    @abstractmethod
    def append(self, frame: np.ndarray) -> None:
        """Append a frame to the stream."""

    @abstractmethod
    def is_active(self) -> bool:
        """Return True if stream is active."""

    @abstractmethod
    def flush(self) -> None:
        """Flush pending stream writes to disk."""

    def _normalize_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this stream type is available (has needed imports)."""

    def __enter__(self) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        """Exit the runtime context related to this object."""
        if self.is_active():
            self.flush()


class MultiPositionOMEStream(OMEStream):
    """Base class for OME streams supporting multi-position imaging datasets.

    Extends OMEStream to handle complex multi-dimensional acquisitions with
    position ('p') dimensions. This class automatically manages index mapping
    and coordinates between global frame indices and backend-specific array
    keys and dimensional indices.

    Provides a default append() implementation that handles multi-position
    data routing, while requiring subclasses to implement `_write_to_backend()`
    for their specific storage mechanisms. The class separates position
    dimensions from other dimensions (time, z, channel, y, x) and creates
    appropriate indexing schemes for efficient data organization.
    """

    def __init__(self) -> None:
        # dimension info for position dimension, if any
        self._position_dim: Dimension | None = None
        # dimension info for other positional dimensions (like grid), if any
        self._positional_dims: list[Dimension] = []
        # Mapping of indices to
        # (array_key, non-position index, position-relative image index)
        self._indices: dict[int, tuple[str, tuple[int, ...], int]] = {}
        # number of times append() has been called
        self._append_count = 0
        # number of positions in the stream
        self._num_positions = 0
        # non-position dimensions
        # (e.g. time, z, c, y, x) that are not positional
        self._non_position_dims: Sequence[Dimension] = []

    def _init_positions(
        self, dimensions: Sequence[Dimension]
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize position tracking and return num_positions, non_position_dims.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            The number of positions and the non-position dimensions.
        """
        # Standard dimensions that don't affect array organization
        standard_dims = {"x", "y", "t", "c", "z"}

        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        # Find non-standard dimensions (excluding position) for positional organization
        positional_dims = [
            d for d in dimensions if d.label not in standard_dims and d.label != "p"
        ]
        # Standard processing dimensions (t, c, z - excluding spatial x, y)
        non_position_dims = [
            d for d in dimensions if d.label in standard_dims and d.label != "p"
        ]

        # Create ranges for non-spatial standard dimensions (for indexing)
        non_p_ranges = [range(d.size) for d in non_position_dims if d.label not in "yx"]

        num_positions = position_dims[0].size if position_dims else 1

        # Create ranges for all positional dimensions in consistent order
        positional_ranges = [range(d.size) for d in positional_dims]

        # Create the cartesian product for all combinations
        # Order: position (if exists), then positional dims, then processing dims
        if position_dims:
            range_iter = enumerate(
                product(range(num_positions), *positional_ranges, *non_p_ranges)
            )
        else:
            range_iter = enumerate(product(*positional_ranges, *non_p_ranges))

        self._position_dim = position_dims[0] if position_dims else None
        self._positional_dims = positional_dims

        # Track position-relative image index for multi-position acquisitions
        position_image_counters: dict[int, int] = {}

        # Create array keys with format "_p000_g000_r000" etc.
        self._indices = {}
        for i, values in range_iter:
            array_key_parts = []

            if position_dims:
                pos = values[0]
                array_key_parts.append(f"_p{pos:04d}")
                remaining_values = values[1:]
            else:
                pos = 0  # Default to position 0 for non-multi-position
                remaining_values = values

            # Add positional dimension parts
            for j, dim in enumerate(positional_dims):
                if j < len(remaining_values):
                    val = remaining_values[j]
                    array_key_parts.append(f"_{dim.label}{val:04d}")

            # Calculate the index for non-positional dimensions
            positional_count = len(positional_dims)
            if position_dims:
                positional_count += 1
            idx = remaining_values[len(positional_dims) :] if remaining_values else []

            # Create the final array key
            if array_key_parts:
                array_key = "".join(array_key_parts)
            elif len(values) == 0:
                array_key = "0"
            else:
                # Fallback for when we have no positional dims
                array_key = "0"

            # Backward compatibility: if only position dimension exists,
            # use simple format (just the position number)
            if position_dims and not positional_dims and values:
                array_key = str(values[0])

            # Get and increment position-relative image index
            image_idx = position_image_counters.get(pos, 0)
            position_image_counters[pos] = image_idx + 1

            self._indices[i] = (array_key, tuple(idx), image_idx)

        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

        # Backward compatibility properties for grid dimension
        grid_dims = [d for d in positional_dims if d.label == "g"]
        self._grid_dim = grid_dims[0] if grid_dims else None
        self._num_grids = grid_dims[0].size if grid_dims else 0

        return num_positions, non_position_dims

    @abstractmethod
    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Backend-specific write implementation.

        Parameters
        ----------
        array_key : str
            The key for the position in the backend (e.g., Zarr group).
        index : tuple[int, ...]
            The index for the non-position dimensions.
        frame : np.ndarray
            The frame data to write.

        Raises
        ------
        RuntimeError
            If the stream is not active or uninitialized.
        """

    def append(self, frame: np.ndarray) -> None:
        if not self.is_active():
            msg = "Stream is closed or uninitialized. Call create() first."
            raise RuntimeError(msg)
        array_key, index, _ = self._indices[self._append_count]
        self._write_to_backend(array_key, index, frame)
        self._append_count += 1
