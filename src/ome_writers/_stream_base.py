from __future__ import annotations

import abc
from abc import abstractmethod
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    import numpy as np

    from ._dimensions import Dimension


STANDARD_DIMS = {"x", "y", "t", "c", "z"}


class FrameIndex(NamedTuple):
    """Index information for a frame in a multi-dimensional acquisition.

    Attributes
    ----------
    array_key : str
        The key identifying which array/file this frame belongs to.
        e.g, "_p0000" for position 0
        e.g, "_p0000_g0001" for position 0, grid 1
        e.g. "_p0000_g0001_r0002" for position 0, grid 1, r 2
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
        - "_p0000_g0001" -> "0:1"
        - "_p0001" -> "1"
        - "0" -> "0"
        """
        # Extract position indices from array_key like "_p0000_g0001_r0002"
        parts = [p for p in self.array_key.split("_") if p]

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

    def __init__(self) -> None:
        # dimension info for position dimension, if any
        self._position_dim: Dimension | None = None
        # Mapping of frame indices to FrameIndex objects
        self._indices: dict[int, FrameIndex] = {}
        # number of times append() has been called
        self._append_count = 0
        # number of positions in the stream
        self._num_positions = 0
        # non-position dimensions (e.g. time, z, c, y, x)
        self._non_position_dims: Sequence[Dimension] = []

    def _init_positions(
        self, dimensions: Sequence[Dimension]
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize position tracking and return num_positions, non_position_dims.

        This method separates dimensions into two categories:
        1. Standard dimensions (x, y, t, c, z) - used for array indexing
        2. Position dimension (p) - for multi-position acquisitions

        The position dimension is used to create separate arrays with keys
        like "_p0000", "_p0001", etc., while standard dimensions are used
        for indexing within each array.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            The number of positions and the non-position dimensions.
        """
        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        # Standard processing dimensions (all non-position dims)
        non_position_dims = [d for d in dimensions if d.label != "p"]

        # Create ranges for non-spatial standard dimensions (for indexing)
        non_p_ranges = [range(d.size) for d in non_position_dims if d.label not in "yx"]

        # get number of positions (first dimension if multiple position_dims or 1)
        num_positions = position_dims[0].size if position_dims else 1

        # Create the cartesian product for all combinations
        # Order: position (if exists), then processing dims
        if position_dims:
            range_iter = enumerate(product(range(num_positions), *non_p_ranges))
            # When there's a position dimension, extract pos from the product
            self._indices = {}
            for i, (pos, *idx) in range_iter:
                array_key = str(pos)
                self._indices[i] = FrameIndex(array_key, tuple(idx))
        else:
            range_iter = enumerate(product(*non_p_ranges))
            # When there's no position dimension, all data goes to array "0"
            self._indices = {}
            for i, idx_tuple in range_iter:
                self._indices[i] = FrameIndex("0", idx_tuple)

        self._position_dim = position_dims[0] if position_dims else None

        # Create array keys as simple position indices: "0", "1", "2", etc.
        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

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
        frame_idx = self._indices[self._append_count]
        self._write_to_backend(frame_idx.array_key, frame_idx.dim_index, frame)
        self._append_count += 1
