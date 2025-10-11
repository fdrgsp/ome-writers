from __future__ import annotations

import importlib
import importlib.util
import threading
import warnings
from collections.abc import Iterator
from contextlib import suppress
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

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
        # Additional positional dimensions (g, r, etc.) for multi-axis support
        self._positional_dims: list[Dimension] = []
        self._is_active = False

    # ------------------------PUBLIC METHODS------------------------ #

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

        # Get unique array keys from tuples
        unique_array_keys = {array_key for array_key, _ in self._indices.values()}
        fnames = self._prepare_files(self._path, sorted(unique_array_keys), overwrite)

        # Create a memmap for each array key
        for array_key, fname in zip(sorted(unique_array_keys), fnames, strict=True):
            # Convert array_key to image_id for metadata
            image_id = self._array_key_to_image_id(array_key)
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
        array_key, dim_index = self._indices[self._append_count]
        self._write_to_backend(array_key, dim_index, frame)
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

    def _create_array_key(
        self,
        positional_values: tuple[int, ...],
        positional_dims: list[Dimension],
        position_dims: list[Dimension],
    ) -> str:
        """Create descriptive array keys for TIFF files with leading underscore.

        TIFF always uses descriptive format (e.g., "_p0000") since files are in a
        flat directory structure, unlike Zarr which uses folders.

        Parameters
        ----------
        linear_index : int
            The linear index from enumerate (not used for TIFF)
        positional_values : tuple[int, ...]
            The values for all positional dimensions (p, g, r, etc.)
        positional_dims : list[Dimension]
            The positional dimensions (g, r, etc., excluding 'p')
        position_dims : list[Dimension]
            The position dimension ('p') if present

        Returns
        -------
        str
            Descriptive array key (e.g., "_p0000_g0001_r0002")
        """
        if not positional_values:
            return "_0"

        # Always use descriptive format for TIFF files
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

        # Create the final array key with leading underscore
        base_key = "_".join(array_key_parts) if array_key_parts else "0"
        return f"_{base_key}"

    def _array_key_to_image_id(self, array_key: str) -> str:
        """Convert array_key to image_id format for OME metadata.

        Examples
        --------
        - "_p0000_g0001" -> "0:1"
        - "_p0001" -> "1"
        - "_0" -> "0"
        """
        # Remove leading underscore
        key = array_key.lstrip("_")

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
            # Convert array_key to image_id for metadata
            image_id = self._array_key_to_image_id(array_key)

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
