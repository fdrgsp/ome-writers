from __future__ import annotations

import abc
from abc import abstractmethod
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self
from yaozarrs.v05 import FieldOfView, Well

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    import numpy as np
    from yaozarrs.v05 import Plate

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
        plate: Plate | None = None,
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
        plate : Plate | None
            Optional ngff plate metadata for HCS data.
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
        # A mapping of indices to (array_key, non-position index)
        self._indices: dict[int, tuple[str, tuple[int, ...]]] = {}
        # number of times append() has been called
        self._append_count = 0
        # number of positions in the stream
        self._num_positions = 0
        # non-position dimensions
        # (e.g. time, z, c, y, x) that are not
        self._non_position_dims: Sequence[Dimension] = []
        # HCS metadata for NGFF structure
        self._plate: Plate | None = None
        self._wells: dict[str, Well] = {}

    def _init_positions(
        self,
        dimensions: Sequence[Dimension],
        plate: Plate | None = None,
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize position tracking and return num_positions, non_position_dims.

        Parameters
        ----------
        dimensions : Sequence[Dimension]
            The dimension information.
        plate : Plate | None
            Optional plate metadata for HCS data.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            The number of positions and the non-position dimensions.
        """
        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        non_position_dims = [d for d in dimensions if d.label != "p"]
        num_positions = position_dims[0].size if position_dims else 1

        # Store HCS metadata
        self._plate = plate
        if plate:
            self._wells = self._extract_wells_from_plate(plate, num_positions)
        else:
            self._wells = {}
        non_p_ranges = [range(d.size) for d in non_position_dims if d.label not in "yx"]
        range_iter = enumerate(product(range(num_positions), *non_p_ranges))

        self._position_dim = position_dims[0] if position_dims else None

        # Create position mapping: for HCS use well paths, otherwise use simple indices
        if self._is_hcs_data():
            # Map position indices to well paths for HCS data with multi-FOV support
            self._indices = {}
            for i, (pos, *idx) in range_iter:
                array_key = self._get_hcs_array_key(pos)
                self._indices[i] = (array_key, tuple(idx))
        else:
            # Use simple position indices for non-HCS data
            self._indices = {i: (str(pos), tuple(idx)) for i, (pos, *idx) in range_iter}

        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

        return num_positions, non_position_dims

    def _extract_wells_from_plate(
        self, plate: Plate, num_positions: int
    ) -> dict[str, Well]:
        """Extract wells dictionary from plate metadata.

        This creates a wells dictionary properly distributing positions/FOVs
        across all wells in the plate.

        Parameters
        ----------
        plate : Plate
            The plate metadata containing well information.
        num_positions : int
            Total number of positions in the acquisition.

        Returns
        -------
        dict[str, Well]
            Dictionary mapping well paths to well metadata.
        """
        wells_dict = {}
        num_wells = len(plate.plate.wells)

        if num_wells == 0:
            return wells_dict

        # Calculate fields per well (distribute positions evenly across wells)
        # TODO: find a better way, can we have different number of fields per well?
        fields_per_well = num_positions // num_wells
        remaining_positions = num_positions % num_wells

        current_field_idx = 0
        for i, well_in_plate in enumerate(plate.plate.wells):
            # Some wells get an extra field if positions don't divide evenly
            extra_field = 1 if i < remaining_positions else 0
            num_fields_this_well = fields_per_well + extra_field

            # Create FieldOfView for each field in this well
            images = []
            for _ in range(num_fields_this_well):
                images.append(FieldOfView(path=str(current_field_idx)))
                current_field_idx += 1

            wells_dict[well_in_plate.path] = Well(images=images, version="0.5")

        return wells_dict

    def _is_hcs_data(self) -> bool:
        """Check if this is HCS data (has both plate and wells metadata)."""
        return self._plate is not None and bool(self._wells)

    def _get_hcs_array_key(self, position_idx: int) -> str:
        """Get the array key for HCS data with multi-FOV support.

        For HCS data, returns well_path/field_idx (e.g., "A/1/0", "A/1/1").
        Position indices map to wells and fields based on the wells metadata.
        """
        # Calculate fields per well from wells metadata
        well_paths = list(self._wells.keys())
        if not well_paths:
            return str(position_idx)

        # Get the number of fields for each well
        fields_per_well = {}
        for well_path, well_meta in self._wells.items():
            fields_per_well[well_path] = len(well_meta.images)

        # Find which well and field this position maps to
        current_pos = 0
        for well_path in well_paths:
            num_fields = fields_per_well[well_path]
            if position_idx < current_pos + num_fields:
                field_idx = position_idx - current_pos
                return f"{well_path}/{field_idx}"
            current_pos += num_fields

        # Fallback to simple index if mapping fails
        return str(position_idx)

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
        array_key, index = self._indices[self._append_count]
        self._write_to_backend(array_key, index, frame)
        self._append_count += 1
