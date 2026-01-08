from __future__ import annotations

import importlib
import importlib.util
import threading
import warnings
from contextlib import suppress
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from typing_extensions import Self

from ome_writers._stream_base import MultiPositionOMEStream


class _WriterParams(TypedDict):
    """Type for writer parameters dictionary."""

    path: str
    shape: tuple[int, ...]
    dtype: Any  # np.dtype
    ome_xml: str


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
    shape is known ahead of time. It supports two writing strategies:

    1. **Thread-based writer (memmap=False, default)**: Uses a background thread with
       a queue to write frames sequentially. This is efficient for streaming writes
       and uses minimal RAM.

    2. **Memmap-based writer (memmap=True)**: Uses numpy.memmap to write frames
       directly to a temporary file, then converts to final OME-TIFF on flush.
       This approach allows random-access writes and can be more memory-efficient
       for very large acquisitions.

    If a 'p' (position) dimension is included in the dimensions, a separate
    OME-TIFF file will be created for each position. In multi-position acquisitions,
    the first position file will contain the complete OME metadata for all positions,
    while subsequent position files will contain a BinaryOnly reference to the main
    file.

    Parameters
    ----------
    flush_interval : int, optional
        For memmap mode: number of frames to acquire before flushing memmaps to disk.
        Default is 100. Ignored in thread mode.

    Attributes
    ----------
    _writers : Dict[int, Union[WriterThread, MemmapWriter]]
        A dictionary mapping position index to its writer (thread or memmap).
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if the tifffile package is available."""
        return bool(
            importlib.util.find_spec("tifffile") is not None
            and importlib.util.find_spec("ome_types") is not None
        )

    def __init__(self, flush_interval: int = 100) -> None:
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
        self._writers: dict[int, WriterThread | MemmapWriter] = {}
        self._queues: dict[int, Queue[np.ndarray | None]] = {}
        self._is_active = False
        self._flush_interval = flush_interval
        self._use_memmap = False

        # Multi-position metadata management
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
        memmap: bool = False,
        **kwargs: Any,
    ) -> Self:
        # Initialize dimensions from MultiPositionOMEStream
        # NOTE: Data will be stored in acquisition order.
        self._init_dimensions(dimensions)

        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        self._use_memmap = memmap
        shape_5d = tuple(d.size for d in self.storage_order_dims)

        fnames = self._prepare_files(self._path, self.num_positions, overwrite)
        ome_xml_list = self._prepare_metadata_for_positions(
            self.storage_order_dims, dtype, fnames, self.num_positions
        )

        # Create a writer for each position
        for p_idx, (fname, ome_xml) in enumerate(
            zip(fnames, ome_xml_list, strict=True)
        ):
            # Shared parameters for both writer types
            writer_params: _WriterParams = {
                "path": fname,
                "shape": shape_5d,
                "dtype": dtype,
                "ome_xml": ome_xml,
            }

            if memmap:
                self._writers[p_idx] = MemmapWriter(
                    **writer_params,
                    flush_interval=self._flush_interval,
                )
            else:
                q = Queue()  # type: ignore
                self._queues[p_idx] = q
                self._writers[p_idx] = thread = WriterThread(
                    **writer_params,
                    image_queue=q,
                )
                thread.start()

        self._is_active = True
        return self

    def is_active(self) -> bool:
        """Return True if the stream is currently active."""
        return self._is_active

    def flush(self) -> None:
        """Flush all pending writes to the underlying TIFF files."""
        if self._use_memmap:
            # Flush all memmap writers to disk and convert to OME-TIFF
            for writer in self._writers.values():
                writer.flush_to_tiff(self._tf)  # type: ignore
        else:
            # Signal the threads to stop by putting None in each queue
            for queue in self._queues.values():
                queue.put(None)

            # Wait for the threads to finish
            for thread in self._writers.values():
                thread.join(timeout=5)  # type: ignore

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

        for position_idx in self._writers:
            self._update_position_metadata(position_idx, metadata)

    # -----------------------PRIVATE METHODS------------------------ #

    def _get_path_root_and_extension(self, path: Path | str) -> tuple[str, str]:
        """Split the path into root and extension for .(ome.)tif and .(ome.)tiff files.

        Returns
        -------
        tuple[str, str]
            A tuple containing the root path (without extension) and the extension.
        """
        path_root = str(path)
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if path_root.endswith(possible_ext):
                return path_root[: -len(possible_ext)], possible_ext
        return path_root, Path(path).suffix

    def _prepare_files(
        self, path: Path, num_positions: int, overwrite: bool
    ) -> list[str]:
        path_root, ext = self._get_path_root_and_extension(path)
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

    def _prepare_metadata_for_positions(
        self,
        tczyx_dims: Sequence[Dimension],
        dtype: np.dtype,
        fnames: list[str],
        num_positions: int,
    ) -> list[str]:
        """Prepare OME-XML metadata for all positions.

        Returns a list of OME-XML strings, one for each position.
        For multi-position acquisitions, the first position file contains
        complete OME metadata for all positions, while subsequent files
        contain BinaryOnly references to the main file.
        """
        # Single position: file gets its own metadata
        if num_positions == 1:
            from ome_writers._dimensions import dims_to_ome

            return [
                dims_to_ome(tczyx_dims, dtype=dtype, tiff_file_name=fname).to_xml()
                for fname in fnames
            ]

        # Multi-position: first file gets complete metadata, others get BinaryOnly
        complete_ome = _create_multiposition_ome(tczyx_dims, dtype, fnames)

        # Store UUID from first image for BinaryOnly references
        if complete_ome.images and complete_ome.images[0].pixels.tiff_data_blocks:
            tiff_data = complete_ome.images[0].pixels.tiff_data_blocks[0]
            if tiff_data.uuid:
                self._main_file_uuid = tiff_data.uuid.value
                self._main_file_name = Path(fnames[0]).name

        # First position gets complete metadata, others get BinaryOnly references
        ome_xml_list = [complete_ome.to_xml()]

        for _ in range(1, num_positions):
            if not self._main_file_uuid or not self._main_file_name:
                raise ValueError(
                    "Main file UUID and Name not set, "
                    "cannot create BinaryOnly reference."
                )
            binary_only_ome = _create_binary_only_ome(
                self._main_file_name, self._main_file_uuid
            )
            ome_xml_list.append(binary_only_ome.to_xml())

        return ome_xml_list

    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TIFF-specific write implementation.

        For thread mode, frames are written sequentially via queue.
        For memmap mode, frames are written directly to the memmap at the given index.
        """
        p_idx = int(position_key)
        if self._use_memmap:
            self._writers[p_idx].write_frame(index, frame)  # type: ignore
        else:
            self._queues[p_idx].put(frame)

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Add OME metadata to TIFF file efficiently without rewriting image data."""
        writer = self._writers[position_idx]
        if not Path(writer._path).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for position {position_idx} does not exist at "
                f"{writer._path}. Not writing metadata.",
                stacklevel=2,
            )
            return

        # Parse the current OME metadata to preserve image names and tiff_data_blocks
        current_metadata = self._ome.OME.from_xml(writer._ome_xml)

        try:
            # Multi-position: position > 0 uses BinaryOnly reference
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
                # For position 0 in multi-position or single position,
                # create position-specific metadata
                is_main_file = self.num_positions > 1 and position_idx == 0
                position_ome = _create_position_specific_ome(
                    position_idx, current_metadata, metadata, is_main_file
                )

            xml = position_ome.to_xml()
            ascii_xml = xml.replace("µ", "&#x00B5;").encode("ascii")

        except Exception as e:
            raise RuntimeError(
                f"Failed to create position-specific OME metadata for position "
                f"{position_idx}. {e}"
            ) from e

        try:
            # TODO:
            # consider a lock on the tiff file itself to prevent concurrent writes?
            self._tf.tiffcomment(writer._path, comment=ascii_xml)
        except Exception as e:
            raise RuntimeError(
                f"Failed to update OME metadata in {writer._path}"
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


class MemmapWriter:
    """Memmap-based writer for a single OME-TIFF position.

    Writes frames directly to a memory-mapped temporary file, then converts
    to final OME-TIFF format on flush. This approach minimizes RAM usage.
    """

    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        ome_xml: str = "",
        flush_interval: int = 100,
        pixelsize: float = 1.0,
    ) -> None:
        import numpy as np

        self._path: str = path
        self._shape = shape
        self._dtype = dtype
        self._ome_xml = ome_xml
        self._flush_interval = flush_interval
        self._res = 1 / pixelsize
        self._frame_count = 0

        # Create temporary memmap file
        self._memmap_path = Path(f"{path}.memmap.{next(memmap_counter)}")
        self._memmap = np.memmap(
            str(self._memmap_path),
            dtype=self._dtype,
            mode="w+",
            shape=self._shape,
            order="C",
        )

    def write_frame(self, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write a single frame to the memmap at the specified index."""
        # place 2D frame into memmap at storage_idx
        idx = (*index, slice(None), slice(None))
        self._memmap[idx] = frame  # type: ignore

        # Periodic flush for durability without per-frame overhead
        self._frame_count += 1
        if self._frame_count % self._flush_interval == 0:
            try:
                self._memmap.flush()
            except Exception:
                pass

    def flush_to_tiff(self, tifffile_module: Any) -> None:
        """Convert memmap to final OME-TIFF file."""
        # Ensure memmap is flushed to disk
        self._memmap.flush()

        # Write memmap array directly to TIFF; tifffile will read from the memmap
        tifffile_module.imwrite(
            self._path,
            self._memmap,
            bigtiff=True,
            ome=False,
            resolutionunit=tifffile_module.RESUNIT.MICROMETER,
            photometric=tifffile_module.PHOTOMETRIC.MINISBLACK,
            description=self._ome_xml,
        )

        # Clean up memmap
        del self._memmap
        self._memmap_path.unlink(missing_ok=True)


memmap_counter = count()

# ------------------------

# helpers for position-specific OME metadata updates


def _create_position_specific_ome(
    position_idx: int,
    current_ome: ome.OME,
    metadata: ome.OME,
    is_main_file: bool = False,
) -> ome.OME:
    """Create OME metadata for a specific position from complete metadata.

    Always preserves tiff_data_blocks (UUID references) and image names.

    Parameters
    ----------
    position_idx : int
        The position index to extract metadata for
    current_ome : ome.OME
        The current OME metadata with original image names and tiff_data_blocks
    metadata : ome.OME
        The new OME metadata to be updated
    is_main_file : bool, optional
        If True, processes ALL images in metadata (for multi-position, position 0).
        If False, extracts only the single image for the given position_idx.
        Default is False.

    Returns
    -------
    ome.OME
        Updated OME metadata with preserved tiff_data_blocks and image names
    """
    if is_main_file:
        # Multi-position mode (position 0): preserve ALL images' tiff_data
        return _preserve_tiff_data(current_ome, metadata)
    else:
        # Standard mode: extract single position's metadata
        target_image_id = f"Image:{position_idx}"

        # Find the image by its ID in the given list of images
        position_image = next(
            img for img in metadata.images if img.id == target_image_id
        )

        # Extract only the relevant image and plates for this position
        position_metadata = ome.OME(
            uuid=metadata.uuid,
            images=[position_image],
            instruments=metadata.instruments,
            plates=_extract_position_plates(metadata, target_image_id),
        )
        # Use the same preservation logic as main file mode
        return _preserve_tiff_data(current_ome, position_metadata)


def _preserve_tiff_data(current_ome: ome.OME, updated_ome: ome.OME) -> ome.OME:
    """Preserve image names and tiff_data_blocks from current metadata.

    This function takes the updated metadata and copies the tiff_data_blocks from the
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


def _create_multiposition_ome(
    tczyx_dims: Sequence[Dimension],
    dtype: np.dtype,
    filenames: list[str],
) -> ome.OME:
    """Create OME metadata for multiple positions with their respective filenames.

    This creates complete metadata for multi-position acquisitions that
    references all position files.

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
    channels = [
        m.Channel(
            id=f"Channel:{i}",
            name=f"Channel {i + 1}",
            samples_per_pixel=1,
        )
        for i in range(dims_sizes.get("c", 1))
    ]

    # Create an image for each position
    for p, filename in enumerate(filenames):
        planes: list[m.Plane] = []
        tiff_blocks: list[m.TiffData] = []
        ifd = 0

        # Generate a unique UUID for each position
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

    ome_obj = m.OME(images=images, creator=f"ome_writers v{__version__}")
    return ome_obj


def _create_binary_only_ome(metadata_file: str, uuid: str) -> ome.OME:
    """Create an OME object with only a BinaryOnly element.

    This is used in multi-position acquisitions to reference the main metadata file
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
