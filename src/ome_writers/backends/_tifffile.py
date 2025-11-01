from __future__ import annotations

import importlib
import importlib.util
import threading
import warnings
from contextlib import suppress
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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
    OME-TIFF file will be created for each position. For multi-position
    acquisitions, the first position file will contain the complete OME metadata
    for all positions, while subsequent position files will contain a BinaryOnly
    reference to the main file.

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
        self._is_active = False
        self._main_file_uuid: str | None = None
        self._main_file_name: str | None = None

    # ------------------------PUBLIC METHODS------------------------ #

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # Initialize dimensions from MultiPositionOMEStream
        # NOTE: since OME-TIFF can store data in any order, we do not enforce
        # OME-NGFF order here.
        self._init_dimensions(dimensions, enforce_ome_order=False)

        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        shape_5d = tuple(d.size for d in self.storage_order_dims)

        fnames = self._prepare_files(self._path, self.num_positions, overwrite)

        # Generate OME XML for each position
        ome_xml_map = _generate_ome_xml_list(
            self.storage_order_dims, dtype, fnames, self.num_positions
        )

        # Set UUID and filename from first position for BinaryOnly references
        if self.num_positions >= 1:
            self._set_main_uuid_and_filename(ome_xml_map[0], fnames[0])

        # Create a memmap for each position
        for p_idx, fname in enumerate(fnames):
            self._queues[p_idx] = q = Queue()  # type: ignore
            self._threads[p_idx] = thread = WriterThread(
                fname,
                shape=shape_5d,
                dtype=dtype,
                image_queue=q,
                ome_xml=ome_xml_map[p_idx],
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
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TIFF-specific write implementation.

        For TIFF, frames are written sequentially, so the index is not used.
        """
        self._queues[int(position_key)].put(frame)

    def _set_main_uuid_and_filename(self, ome_xml: str, fname: str) -> None:
        """Set main file UUID and name for BinaryOnly references."""
        first_ome = self._ome.OME.from_xml(ome_xml)
        if first_ome.images and first_ome.images[0].pixels.tiff_data_blocks:
            tiff_data = first_ome.images[0].pixels.tiff_data_blocks[0]
            if tiff_data.uuid:
                self._main_file_uuid = tiff_data.uuid.value
                self._main_file_name = Path(fname).name

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Add OME metadata to TIFF file efficiently without rewriting image data.

        For multi-position acquisitions:
        - Position 0 gets complete metadata for all positions
        - Positions > 0 get BinaryOnly references to the first file
        """
        thread = self._threads[position_idx]
        if not Path(thread._path).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for position {position_idx} does not exist at "
                f"{thread._path}. Not writing metadata.",
                stacklevel=2,
            )
            return

        # Parse the current OME metadata to preserve image names and tiff_data_blocks
        current_metadata = self._ome.OME.from_xml(thread._ome_xml)

        try:
            # For multi-position with position > 0, use BinaryOnly reference
            if self.num_positions > 1 and position_idx > 0:
                if not self._main_file_uuid or not self._main_file_name:
                    msg = (
                        "Main file UUID and name not set. "
                        "Cannot create BinaryOnly reference."
                    )
                    raise ValueError(msg)
                position_ome = _create_binary_only_ome(
                    self._main_file_name, self._main_file_uuid
                )
            else:
                # For single position or first file in multi-position,
                # preserve all tiff_data from current metadata
                position_ome = _preserve_tiff_data(current_metadata, metadata)

            # Create ASCII version for tifffile.tiffcomment since tifffile.tiffcomment
            # requires ASCII strings
            ascii_xml = position_ome.to_xml().replace("µ", "&#x00B5;").encode("ascii")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create position-specific OME metadata for position "
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

# helpers for OME metadata updates


def _generate_ome_xml_list(
    dims: Sequence[Dimension],
    dtype: np.dtype,
    filenames: list[str],
    num_positions: int,
) -> dict[int, str]:
    """Generate OME XML for each position file.

    For single position: creates standard OME metadata.
    For multi-position: first file gets complete metadata, others get BinaryOnly.

    Parameters
    ----------
    dims : Sequence[Dimension]
        The dimensions for the data (t, c, z, y, x)
    dtype : np.dtype
        The data type of the images
    filenames : list[str]
        List of filenames for each position
    num_positions : int
        Number of positions

    Returns
    -------
    dict[int, str]
        Dictionary mapping position index to OME-XML string
    """
    if num_positions == 1:
        # Single position: standard OME metadata
        from ome_writers._dimensions import dims_to_ome

        ome_obj = dims_to_ome(dims, dtype=dtype, tiff_file_name=filenames[0])
        return {0: ome_obj.to_xml()}

    # Multi-position: first file gets complete metadata, others get BinaryOnly
    complete_ome = _create_multiposition_ome(dims, dtype, filenames)

    # Extract UUID from first image for BinaryOnly references
    main_file_uuid: str | None = None
    main_file_name: str | None = None
    if complete_ome.images and complete_ome.images[0].pixels.tiff_data_blocks:
        tiff_data = complete_ome.images[0].pixels.tiff_data_blocks[0]
        if tiff_data.uuid:
            main_file_uuid = tiff_data.uuid.value
            main_file_name = Path(filenames[0]).name

    # First position gets complete metadata, others get BinaryOnly references
    ome_xml_map = {0: complete_ome.to_xml()}
    for p_idx in range(1, num_positions):
        if not main_file_uuid or not main_file_name:
            raise ValueError(
                "Main file UUID and Name not set, cannot create BinaryOnly reference."
            )
        binary_only_ome = _create_binary_only_ome(main_file_name, main_file_uuid)
        ome_xml_map[p_idx] = binary_only_ome.to_xml()

    return ome_xml_map


def _preserve_tiff_data(current_ome: ome.OME, updated_ome: ome.OME) -> ome.OME:
    """Preserve tiff_data_blocks and image names when updating OME metadata.

    This function updates the OME metadata while preserving tiff_data_blocks from the
    current metadata to preserve file references.

    Parameters
    ----------
    current_ome : ome.OME
        The current OME metadata with original image names and tiff_data_blocks
    updated_ome : ome.OME
        The new OME metadata to be updated

    Returns
    -------
    ome.OME
        Updated metadata with preserved image names and tiff_data_blocks
    """
    # Create a mapping of current images by their ID
    current_images_map = {img.id: img for img in current_ome.images}

    # Update each image in the new metadata
    updated_images = []
    for new_image in updated_ome.images:
        current_image = current_images_map.get(new_image.id)
        if current_image is not None:
            # Preserve the image name and tiff_data_blocks
            updated_pixels = _copy_tiff_data_blocks(
                current_image.pixels, new_image.pixels
            )
            updated_image = new_image.model_copy(update={"pixels": updated_pixels})
            updated_images.append(updated_image)
        else:
            updated_images.append(new_image)

    return updated_ome.model_copy(update={"images": updated_images})


def _copy_tiff_data_blocks(
    source_pixels: ome.Pixels | None, destination_pixels: ome.Pixels | None
) -> ome.Pixels | None:
    if (
        destination_pixels is None
        or source_pixels is None
        or source_pixels.tiff_data_blocks is None
    ):
        return destination_pixels

    # Deep copy the tiff_data_blocks to avoid reference issues
    copied_blocks = [
        block.model_copy(deep=True) for block in source_pixels.tiff_data_blocks
    ]
    return destination_pixels.model_copy(update={"tiff_data_blocks": copied_blocks})


def _create_multiposition_ome(
    tczyx_dims: Sequence[Dimension],
    dtype: np.dtype,
    filenames: list[str],
) -> ome.OME:
    """Create complete OME metadata for multi-position acquisition with all filenames.

    This is used for multi-position acquisitions to create complete metadata that
    references all position files in the first file.

    Parameters
    ----------
    tczyx_dims : Sequence[Dimension]
        The dimensions for the data (t, c, z, y, x)
    dtype : np.dtype
        The data type of the images
    filenames : list[str]
        List of filenames for each position

    Returns
    -------
    ome.OME
        An OME object containing all images with their respective file references
    """
    try:
        from ome_types import model as m
    except ImportError as e:
        raise ImportError(
            "The `ome-types` package is required to use this function. "
            "Please install it via `pip install ome-types` or use the `tiff` extra."
        ) from e

    import uuid

    from ome_writers import __version__

    # Get dimensions info
    dims_sizes = {dim.label: dim.size for dim in tczyx_dims}
    _dim_names = "".join(reversed(dims_sizes)).upper()
    dim_order = next(
        (x for x in m.Pixels_DimensionOrder if x.value.startswith(_dim_names)),
        m.Pixels_DimensionOrder.XYCZT,
    )

    images: list[m.Image] = []

    # Create an image for each position
    for p, filename in enumerate(filenames):
        # Each image needs its own unique channel IDs
        channels = [
            m.Channel(
                id=f"Channel:{p}:{i}",
                name=f"Channel {i + 1}",
                samples_per_pixel=1,
            )
            for i in range(dims_sizes.get("c", 1))
        ]

        planes: list[m.Plane] = []
        tiff_blocks: list[m.TiffData] = []
        ifd = 0
        uuid_ = f"urn:uuid:{uuid.uuid4()}"

        # iterate over ordered cartesian product of tcz sizes
        labels, sizes = zip(
            *[(d.label, d.size) for d in tczyx_dims if d.label in "tcz"], strict=False
        )
        has_z, has_t, has_c = "z" in labels, "t" in labels, "c" in labels
        for index in np.ndindex(*sizes):
            plane = m.Plane(
                the_z=index[labels.index("z")] if has_z else 0,
                the_t=index[labels.index("t")] if has_t else 0,
                the_c=index[labels.index("c")] if has_c else 0,
            )
            planes.append(plane)
            tiff_data = m.TiffData(
                ifd=ifd,
                uuid=m.TiffData.UUID(value=uuid_, file_name=Path(filename).name),
                first_c=plane.the_c,
                first_z=plane.the_z,
                first_t=plane.the_t,
                plane_count=1,
            )
            tiff_blocks.append(tiff_data)
            ifd += 1

        pix_type = m.PixelType(np.dtype(dtype).name)
        pixels = m.Pixels(
            id=f"Pixels:{p}",
            channels=channels,
            planes=planes,
            tiff_data_blocks=tiff_blocks,
            dimension_order=dim_order,
            type=pix_type,
            size_x=dims_sizes.get("x", 1),
            size_y=dims_sizes.get("y", 1),
            size_z=dims_sizes.get("z", 1),
            size_c=dims_sizes.get("c", 1),
            size_t=dims_sizes.get("t", 1),
        )

        base_name = Path(filename).stem
        images.append(
            m.Image(
                id=f"Image:{p}",
                name=base_name,
                pixels=pixels,
            )
        )

    ome = m.OME(images=images, creator=f"ome_writers v{__version__}")
    return ome


def _create_binary_only_ome(metadata_file: str, uuid: str) -> ome.OME:
    """Create a BinaryOnly OME metadata reference.

    This creates minimal OME-XML that references the complete metadata file
    from secondary position files.

    Parameters
    ----------
    metadata_file : str
        The filename of the main metadata file (e.g., "file_p000.ome.tif")
    uuid : str
        The UUID of the main file (e.g., "urn:uuid:...")

    Returns
    -------
    ome.OME
        An OME object containing only a BinaryOnly element
    """
    binary_only = ome.OME.BinaryOnly(metadata_file=metadata_file, uuid=uuid)
    return ome.OME(binary_only=binary_only)
