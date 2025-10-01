from __future__ import annotations

import importlib
import importlib.util
import threading
import uuid
import warnings
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
        # Using dictionaries to handle multi-position ('p') acquisitions
        self._threads: dict[int, WriterThread] = {}
        self._queues: dict[int, Queue[np.ndarray | None]] = {}
        # Store UUIDs for each file for multi-file OME-TIFF compliance
        self._file_uuids: dict[int, str] = {}
        self._file_names: dict[int, str] = {}
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
        num_positions, tczyx_dims = self._init_positions(dimensions)
        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        shape_5d = tuple(d.size for d in tczyx_dims)

        fnames = self._prepare_files(self._path, num_positions, overwrite)

        # Generate unique UUIDs for each file
        for p_idx, fname in enumerate(fnames):
            self._file_uuids[p_idx] = f"urn:uuid:{uuid.uuid4()}"
            self._file_names[p_idx] = Path(fname).name

        # Create a memmap for each position
        for p_idx, fname in enumerate(fnames):
            ome = dims_to_ome(tczyx_dims, dtype=dtype, tiff_file_name=fname)
            self._queues[p_idx] = q = Queue()  # type: ignore
            self._threads[p_idx] = thread = WriterThread(
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

        for position_idx in self._threads:
            self._update_position_metadata(position_idx, metadata)

    # -----------------------PRIVATE METHODS------------------------ #

    def _prepare_files(
        self, path: Path, num_positions: int, overwrite: bool
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
        for p_idx in range(num_positions):
            # only append position index if there are multiple positions
            if num_positions > 1:
                p_path = Path(f"{path_root}_p{p_idx:03d}{ext}")
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
        self._queues[int(array_key)].put(frame)

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Add OME metadata to TIFF file with multi-file OME-TIFF compliance.

        For multi-file OME-TIFF, each file contains the complete OME-XML metadata
        with TiffData elements referencing all files via UUID.
        """
        thread = self._threads[position_idx]
        if not Path(thread._path).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for position {position_idx} does not exist at "
                f"{thread._path}. Not writing metadata.",
                stacklevel=2,
            )
            return

        try:
            # Create complete multi-file OME metadata for this file
            file_ome = _create_multifile_ome_metadata(
                position_idx, metadata, self._file_uuids, self._file_names
            )
            # Create ASCII version for tifffile.tiffcomment since tifffile.tiffcomment
            # requires ASCII strings
            ascii_xml = file_ome.to_xml().replace("µ", "&#x00B5;").encode("ascii")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create multi-file OME metadata for position "
                f"{position_idx}. {e}"
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


def _create_multifile_ome_metadata(
    current_position_idx: int,
    metadata: ome.OME,
    file_uuids: dict[int, str],
    file_names: dict[int, str],
) -> ome.OME:
    """Create complete multi-file OME metadata for a specific file.

    Each file in a multi-file OME-TIFF dataset contains the complete OME-XML
    metadata with TiffData elements that reference all files using UUID child
    elements, as required by the OME-TIFF specification.

    Parameters
    ----------
    current_position_idx : int
        The position index of the current file being written
    metadata : ome.OME
        The complete OME metadata for the entire dataset
    file_uuids : dict[int, str]
        Mapping of position index to UUID for each file
    file_names : dict[int, str]
        Mapping of position index to filename for each file

    Returns
    -------
    ome.OME
        Complete OME metadata with proper multi-file TiffData references
    """
    # Create a copy of the complete metadata but with the current file's UUID
    current_uuid = file_uuids[current_position_idx]

    # Create TiffData elements for all images with proper UUID references
    updated_images = []
    for image in metadata.images:
        # Extract position index from image ID (assumes format "Image:{position_idx}")
        try:
            img_position_idx = int(image.id.split(":")[-1])
        except (ValueError, IndexError):
            # Fallback if image ID doesn't follow expected format
            img_position_idx = 0

        # Create TiffData with UUID reference to the appropriate file
        target_uuid = file_uuids.get(img_position_idx, current_uuid)
        target_filename = file_names.get(
            img_position_idx, file_names[current_position_idx]
        )

        # Create new TiffData elements for this image
        tiff_data_list = []
        if image.pixels and image.pixels.tiff_data_blocks:
            for tiff_data in image.pixels.tiff_data_blocks:
                new_tiff_data = ome.TiffData(
                    ifd=tiff_data.ifd,
                    first_c=tiff_data.first_c,
                    first_t=tiff_data.first_t,
                    first_z=tiff_data.first_z,
                    plane_count=tiff_data.plane_count,
                    uuid=ome.TiffData.UUID(
                        value=target_uuid, file_name=target_filename
                    ),
                )
                tiff_data_list.append(new_tiff_data)

        # Create updated image with new TiffData
        if image.pixels:
            updated_pixels = ome.Pixels(
                id=image.pixels.id,
                dimension_order=image.pixels.dimension_order,
                type=image.pixels.type,
                size_x=image.pixels.size_x,
                size_y=image.pixels.size_y,
                size_z=image.pixels.size_z,
                size_c=image.pixels.size_c,
                size_t=image.pixels.size_t,
                channels=image.pixels.channels,
                tiff_data_blocks=tiff_data_list,
                planes=image.pixels.planes,
            )

            updated_image = ome.Image(
                id=image.id,
                name=image.name,
                pixels=updated_pixels,
            )
            updated_images.append(updated_image)

    # Return complete OME metadata with current file's UUID and all cross-references
    return ome.OME(
        uuid=current_uuid,
        images=updated_images,
        instruments=metadata.instruments,
        plates=metadata.plates,
    )
