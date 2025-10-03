from __future__ import annotations

import importlib
import importlib.util
import threading
import warnings
from contextlib import suppress
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import Self

from ome_writers._dimensions import dims_to_ome
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    import ome_types.model as ome

    from ome_writers._dimensions import Dimension

else:
    with suppress(ImportError):
        import ome_types.model as ome


STANDARD_DIMS = {"x", "y", "t", "c", "z"}


class FrameIndex(NamedTuple):
    """Index information for a frame in a multi-dimensional acquisition.

    This is used specifically by the TIFF backend to support multi-axis
    positional dimensions (p, g, r, etc.) with hierarchical Image IDs.

    Attributes
    ----------
    array_key : str
        The key identifying which array/file this frame belongs to.
        e.g, "_p0000" for position 0
        e.g, "_p0000_g0001" for position 0, grid 1
        e.g. "_p0000_g0001_r0002" for position 0, grid 1, region 2
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


class TifffileStream(MultiPositionOMEStream):
    """A concrete OMEStream implementation for writing to OME-TIFF files.

    This writer is designed for deterministic acquisitions where the full experiment
    shape is known ahead of time. It works by creating all necessary OME-TIFF
    files at the start of the acquisition and using memory-mapped arrays for
    efficient, sequential writing of incoming frames.

    If a 'p' (position) dimension is included in the dimensions, a separate
    OME-TIFF file will be created for each position.

    Attributes
    ----------
    _writers : Dict[int, np.memmap]
        A dictionary mapping position index to its numpy memmap array.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if the tifffile package is available."""
        return bool(
            importlib.util.find_spec("tifffile") is not None
            and importlib.util.find_spec("ome_types") is not None
        )

    def __init__(self) -> None:
        super().__init__()
        try:
            import ome_types.model
            import tifffile
        except ImportError as e:
            msg = (
                "TifffileStream requires tifffile and ome-types: "
                "`pip install ome-writers[tifffile]`."
            )
            raise ImportError(msg) from e

        self._tf = tifffile
        self._ome = ome_types.model
        # Using dictionaries to handle multi-position ('p')
        # and other positional acquisitions
        self._threads: dict[str, WriterThread] = {}
        self._queues: dict[str, Queue[np.ndarray | None]] = {}
        # Mapping from array_key to FrameIndex for metadata operations
        self._array_key_to_frame_idx: dict[str, FrameIndex] = {}
        # Additional positional dimensions (g, r, etc.) for multi-axis support
        self._positional_dims: list[Dimension] = []
        # Override base class type - TIFF uses FrameIndex instead of plain tuples
        self._indices: dict[int, FrameIndex] = {}  # type: ignore[assignment]
        self._is_active = False

    # ------------------------PUBLIC METHODS------------------------ #

    def _init_positions(
        self, dimensions: Sequence[Dimension]
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize position tracking with multi-axis support for TIFF backend.

        This override extends the base implementation to support additional
        positional dimensions (g, r, etc.) beyond the standard 'p' dimension.
        These positional dimensions are used to create hierarchical Image IDs
        in OME metadata (e.g., "Image:0:1" for position 0, grid 1).

        This method separates dimensions into three categories:
        1. Standard dimensions (x, y, t, c, z) - used for array indexing
        2. Position dimension (p) - primary positional dimension
        3. Positional dimensions (g, r, etc.) - additional positional organization

        Positional dimensions (p, g, r, etc.) are used to create separate arrays
        with keys like "_p0000_g0001_r0002", while standard dimensions are used
        for indexing within each array.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            The number of positions and the non-position dimensions.
        """
        from itertools import product

        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        # Find non-standard dimensions (excluding position) for positional organization
        positional_dims = [
            d for d in dimensions if d.label not in STANDARD_DIMS and d.label != "p"
        ]
        # Standard processing dimensions (t, c, z - excluding spatial x, y)
        non_position_dims = [
            d for d in dimensions if d.label in STANDARD_DIMS and d.label != "p"
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

        # Create array keys with format "_p0000_g0000_r0000" etc.
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
            else:
                # No positional dims at all (single acquisition)
                array_key = "0"

            self._indices[i] = FrameIndex(array_key, tuple(idx))

        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

        return num_positions, non_position_dims

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic
        _, tczyx_dims = self._init_positions(dimensions)
        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        shape_5d = tuple(d.size for d in tczyx_dims)

        # Get unique array keys and prepare files
        unique_array_keys = {
            frame_idx.array_key for frame_idx in self._indices.values()
        }
        fnames = self._prepare_files(self._path, sorted(unique_array_keys), overwrite)

        # Create a mapping from array_key to FrameIndex for metadata operations
        self._array_key_to_frame_idx = {
            frame_idx.array_key: frame_idx for frame_idx in self._indices.values()
        }

        # Create a memmap for each array key
        for array_key, fname in zip(sorted(unique_array_keys), fnames, strict=True):
            # Get image_id from FrameIndex
            image_id = self._array_key_to_frame_idx[array_key].image_id
            ome = dims_to_ome(
                tczyx_dims, dtype=dtype, tiff_file_name=fname, image_id=image_id
            )
            self._queues[array_key] = q = Queue()  # type: ignore
            self._threads[array_key] = thread = WriterThread(
                fname,
                shape=shape_5d,
                dtype=dtype,
                image_queue=q,
                ome_xml=ome.to_xml(),
            )
            thread.start()

        self._is_active = True
        return self

    def is_active(self) -> bool:
        """Return True if the stream is currently active."""
        return self._is_active

    def append(self, frame: np.ndarray) -> None:
        """Append a frame to the TIFF stream."""
        if not self.is_active():
            msg = "Stream is closed or uninitialized. Call create() first."
            raise RuntimeError(msg)
        frame_idx = self._indices[self._append_count]  # type: ignore[assignment]
        self._write_to_backend(frame_idx.array_key, frame_idx.dim_index, frame)
        self._append_count += 1

    def flush(self) -> None:
        """Flush all pending writes to the underlying TIFF files."""
        # Signal the threads to stop by putting None in each queue
        for queue in self._queues.values():
            queue.put(None)

        # Wait for the thread to finish
        for thread in self._threads.values():
            thread.join(timeout=5)

        # Mark as inactive after flushing - this is consistent with other backends
        self._is_active = False

    def update_ome_metadata(self, metadata: ome.OME) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be an instance of ome_types.OME.

        This method should be called after flush() to update the OME-XML
        description in the already-written TIFF files with complete metadata.

        Parameters
        ----------
        metadata : OME
            The OME metadata object to write to the TIFF files.
        """
        if not isinstance(metadata, self._ome.OME):  # pragma: no cover
            raise TypeError(f"Expected OME metadata, got {type(metadata)}")

        for array_key in self._threads:
            self._update_position_metadata(array_key, metadata)

    # -----------------------PRIVATE METHODS------------------------ #

    def _prepare_files(
        self, path: Path, array_keys: list[str], overwrite: bool
    ) -> list[str]:
        path_root = str(path)
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if path_root.endswith(possible_ext):
                ext = possible_ext
                path_root = path_root[: -len(possible_ext)]
                break
        else:
            ext = path.suffix

        fnames = []
        for array_key in array_keys:
            # only append array_key if there are multiple arrays
            if len(array_keys) > 1:
                p_path = Path(f"{path_root}{array_key}{ext}")
            else:
                p_path = self._path

            # Check if file exists and handle overwrite logic
            if p_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"File {p_path} already exists. "
                        "Use overwrite=True to overwrite it."
                    )
                p_path.unlink()

            # Ensure the parent directory exists
            p_path.parent.mkdir(parents=True, exist_ok=True)
            fnames.append(str(p_path))

        return fnames

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TIFF-specific write implementation."""
        self._queues[array_key].put(frame)

    def _update_position_metadata(self, array_key: str, metadata: ome.OME) -> None:
        """Add OME metadata to TIFF file efficiently without rewriting image data."""
        thread = self._threads[array_key]
        if not Path(thread._path).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for array key {array_key} does not exist at "
                f"{thread._path}. Not writing metadata.",
                stacklevel=2,
            )
            return

        try:
            # Get the FrameIndex from the mapping
            frame_idx = self._array_key_to_frame_idx[array_key]
            # Use the image_id property from FrameIndex
            image_id = frame_idx.image_id

            # Create position-specific OME metadata using the full image_id
            position_ome = _create_position_specific_ome(image_id, metadata)
            # Create ASCII version for tifffile.tiffcomment since tifffile.tiffcomment
            # requires ASCII strings
            ascii_xml = position_ome.to_xml().replace("µ", "&#x00B5;").encode("ascii")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create position-specific OME metadata for array key "
                f"{array_key}. {e}"
            ) from e

        try:
            # TODO:
            # consider a lock on the tiff file itself to prevent concurrent writes?
            self._tf.tiffcomment(thread._path, comment=ascii_xml)
        except Exception as e:
            raise RuntimeError(
                f"Failed to update OME metadata in {thread._path}"
            ) from e


class WriterThread(threading.Thread):
    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        image_queue: Queue[np.ndarray | None],
        ome_xml: str = "",
        pixelsize: float = 1.0,
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(thread_counter)}")
        self._path: str = path
        self._shape = shape
        self._dtype = dtype
        self._image_queue = image_queue
        self._res = 1 / pixelsize
        self._bytes_written = 0
        self._frames_written = 0
        self._ome_xml = ome_xml

    def run(self) -> None:
        # would be nice if we could just use `iter(queue, None)`...
        # but that doesn't work with numpy arrays which don't support __eq__
        import tifffile

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Generator to yield frames from the queue."""
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                yield frame
                self._bytes_written += frame.nbytes
                self._frames_written += 1

        try:
            # Create TiffWriter and write the data
            # Since we're using tiffcomment for metadata updates,
            # we can close immediately
            with tifffile.TiffWriter(self._path, bigtiff=True, ome=False) as writer:
                writer.write(
                    _queue_iterator(),
                    shape=self._shape,
                    dtype=self._dtype,
                    resolution=(self._res, self._res),
                    resolutionunit=tifffile.RESUNIT.MICROMETER,
                    photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                    description=self._ome_xml,
                )

        except Exception as e:
            # suppress an over-eager tifffile exception
            # when the number of bytes written is less than expected
            if "wrong number of bytes" in str(e):
                return
            raise


thread_counter = count()

# ------------------------

# helpers for position-specific OME metadata updates

def _create_position_specific_ome(image_id: str, metadata: ome.OME) -> ome.OME:
    """Create OME metadata for a specific position from complete metadata.

    Extracts only the Image and related metadata for the given image_id.
    Image IDs follow the pattern "Image:{image_id}" where image_id can be:
    - Simple: "0", "1", "2" (single position)
    - Multi-axis: "0:0", "0:1", "1:0" (position:grid format)
    - Three-axis: "0:0:0", "0:1:2" (position:grid:other format)
    """
    target_image_id = f"Image:{image_id}"

    # Find an image by its ID in the given list of images
    # will raise StopIteration if not found (caller should catch error)
    position_image = next(img for img in metadata.images if img.id == target_image_id)
    position_plates = _extract_position_plates(metadata, target_image_id)

    return ome.OME(
        uuid=metadata.uuid,
        images=[position_image],
        instruments=metadata.instruments,
        plates=position_plates,
    )


def _extract_position_plates(ome: ome.OME, target_image_id: str) -> list[ome.Plate]:
    """Extract plate metadata for a specific image ID.

    Searches through plates to find the well sample referencing the target
    image ID and returns a plate containing only the relevant well and sample.
    """
    for plate in ome.plates:
        for well in plate.wells:
            if _well_contains_image(well, target_image_id):
                return [_create_position_plate(plate, well, target_image_id)]

    return []


def _well_contains_image(well: ome.Well, target_image_id: str) -> bool:
    """Check if a well contains a sample referencing the target image ID."""
    return any(
        sample.image_ref and sample.image_ref.id == target_image_id
        for sample in well.well_samples
    )


def _create_position_plate(
    original_plate: ome.Plate, well: ome.Well, target_image_id: str
) -> ome.Plate:
    """Create a new plate containing only the relevant well and sample."""
    # Find the specific well sample for this image
    target_sample = next(
        sample
        for sample in well.well_samples
        if sample.image_ref and sample.image_ref.id == target_image_id
    )

    # Create new plate with only the relevant well and sample
    plate_dict = original_plate.model_dump()
    well_dict = well.model_dump()
    well_dict["well_samples"] = [target_sample]
    plate_dict["wells"] = [well_dict]
    return ome.Plate.model_validate(plate_dict)
