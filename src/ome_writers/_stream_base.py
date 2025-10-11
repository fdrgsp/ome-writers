from __future__ import annotations

import abc
from abc import abstractmethod
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    import numpy as np

    from ._dimensions import Dimension


class FrameIndex(NamedTuple):
    """Index information for a frame in a multi-dimensional acquisition.

    This is used to support multi-axis positional dimensions (p, g, r, etc.)
    with hierarchical organization and Image IDs for metadata.

    Attributes
    ----------
    array_key : str
        The key identifying which array/file this frame belongs to.
        Examples: "p0000", "p0000_g0001", "p0000_g0001_r0002", "_p0000"
    dim_index : tuple[int, ...]
        The index tuple for non-positional dimensions (t, c, z).
    """

    array_key: str
    dim_index: tuple[int, ...]

    @property
    def image_id(self) -> str:
        """Convert array_key to image_id format for OME metadata.

        Examples
        --------
        - "p0000_g0001" -> "0:1"
        - "_p0000_g0001" -> "0:1" (leading underscore stripped)
        - "p0001" -> "1"
        - "0" -> "0"
        """
        # Remove leading underscore if present (TIFF variant)
        key = self.array_key.lstrip("_")

        # Extract position indices from array_key like "p0000_g0001_r0002"
        parts = [p for p in key.split("_") if p]

        if len(parts) == 1:
            if (name := parts[0]).isdigit():
                return name
            else:
                # keep only numeric characters
                name = "".join(c for c in name if c.isdigit())
                return str(int(name))

        indices = []
        for part in parts:
            if part and len(part) > 1:
                try:
                    indices.append(str(int(part[1:])))
                except ValueError:
                    continue

        return ":".join(indices) if indices else "0"


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

    # Standard OME dimensions that should not be treated as positional
    STANDARD_DIMS: ClassVar[set[str]] = {"x", "y", "t", "c", "z"}

    def __init__(self) -> None:
        # dimension info for position dimension, if any
        self._position_dim: Dimension | None = None
        # A mapping of indices to FrameIndex
        self._indices: dict[int, FrameIndex] = {}
        # number of times append() has been called
        self._append_count = 0
        # number of positions in the stream
        self._num_positions = 0
        # non-position dimensions
        # (e.g. time, z, c, y, x) that are not
        self._non_position_dims: Sequence[Dimension] = []
        # Additional positional dimensions (g, r, etc.) for multi-axis support
        self._positional_dims: list[Dimension] = []

    def _init_positions(
        self, dimensions: Sequence[Dimension]
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize position tracking with multi-axis support.

        This method support additional positional dimensions (g, r, etc.) beyond the
        standard 'p' dimension. These positional dimensions are used to create separate
        arrays with simple numeric keys (0, 1, 2, ...).

        This method separates dimensions into three categories:
        1. Standard dimensions (x, y, t, c, z) - used for array indexing
        2. Position dimension (p) - primary positional dimension
        3. Positional dimensions (g, r, etc.) - additional positional organization

        Positional dimensions (p, g, r, etc.) are used to create separate arrays
        with keys like "0", "1", "2", etc., while standard dimensions are used
        for indexing within each array.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            The number of positions and the non-position dimensions.
        """
        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        # Find non-standard dimensions (excluding position) for positional organization
        positional_dims = [
            d
            for d in dimensions
            if d.label not in self.STANDARD_DIMS and d.label != "p"
        ]
        # Standard processing dimensions (t, c, z - excluding spatial x, y)
        non_position_dims = [
            d for d in dimensions if d.label in self.STANDARD_DIMS and d.label != "p"
        ]

        # Create ranges for non-spatial standard dimensions (for indexing)
        non_p_ranges = [range(d.size) for d in non_position_dims if d.label not in "yx"]

        # get number of positions (first dimension if multiple position_dims or 1)
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

        # Create array keys with simple numeric format: "0", "1", "2", etc.
        self._indices = {}
        for i, values in range_iter:
            # Calculate how many positional values we have
            positional_count = len(positional_dims)
            if position_dims:
                positional_count += 1

            # Extract positional values
            positional_values = (
                values[:positional_count] if positional_count > 0 else ()
            )

            # Create the array key
            array_key = self._create_array_key(
                positional_values, positional_dims, position_dims
            )

            # The remaining values are the standard dimension indices
            idx = values[positional_count:] if positional_count > 0 else values

            # Create index entry
            self._indices[i] = FrameIndex(array_key, tuple(idx))

        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

        return num_positions, non_position_dims

    def _create_array_key(
        self,
        positional_values: tuple[int, ...],
        positional_dims: list[Dimension],
        position_dims: list[Dimension],
    ) -> str:
        """Create an array key from positional values.

        Uses simple numeric keys ("0", "1", "2") when only position dimension exists.
        Uses descriptive keys ("p0000_g0001") when multiple positional axes are present.

        Parameters
        ----------
        positional_values : tuple[int, ...]
            The values for all positional dimensions (p, g, r, etc.)
        positional_dims : list[Dimension]
            The positional dimensions (g, r, etc., excluding 'p')
        position_dims : list[Dimension]
            The position dimension ('p') if present

        Returns
        -------
        str
            The array key for this combination of positional values
        """
        if not positional_values:
            return "0"

        # If we only have position dimension (no other positional axes),
        # use simple numeric format: "0", "1", "2", etc.
        if position_dims and not positional_dims:
            return str(positional_values[0])

        # If we have no position dimension but have other positional dims,
        # use simple numeric format if only one positional dim
        if not position_dims and len(positional_dims) == 1:
            return str(positional_values[0])

        # Multiple positional axes: use descriptive format
        array_key_parts = []

        if position_dims:
            pos = positional_values[0]
            array_key_parts.append(f"p{pos:04d}")
            remaining_values = positional_values[1:]
        else:
            remaining_values = positional_values

        # Add positional dimension parts
        for j, dim in enumerate(positional_dims):
            if j < len(remaining_values):
                val = remaining_values[j]
                array_key_parts.append(f"{dim.label}{val:04d}")

        # Create the final array key
        return "_".join(array_key_parts) if array_key_parts else "0"

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
        frame_idx = self._indices[self._append_count]
        self._write_to_backend(frame_idx.array_key, frame_idx.dim_index, frame)
        self._append_count += 1
