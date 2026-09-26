"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import json
import math
import threading
import warnings
import weakref
from contextlib import suppress
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from ome_writers._backends._backend import ArrayBackend
from ome_writers._backends._ome_xml import prepare_metadata
from ome_writers._backends._tiff_array import FinalizedTiffArray, LiveTiffArray

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from ome_writers._backends._backend import ArrayLike
    from ome_writers._backends._ome_xml import OmeXMLMirror
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings, Dimension

try:
    import ome_types.model as ome
    import tifffile
except ImportError as e:
    raise ImportError(
        f"{__name__} requires tifffile and ome-types: "
        "`pip install ome-writers[tifffile]`."
    ) from e

PLANE_KEYS = {
    "delta_t",
    "exposure_time",
    "position_x",
    "position_y",
    "position_z",
}


@dataclass
class PositionManager:
    """Per-position writer/metadata state for TIFF backend."""

    file_path: str
    write_state: TiffWriteState | None
    metadata_mirror: OmeXMLMirror

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._metadata_dirty: bool = False

    def update_metadata(self, metadata: ome.OME, flush: bool = False) -> None:
        """Update cached metadata and mark as dirty.  Optionally flush to file."""
        with self._lock:
            self.metadata_mirror.model = metadata
            self.metadata_mirror.mark_dirty()

        # careful... our lock is not re-entrant, so avoid deadlock
        self.metadata_mirror.flush(force=flush)

    def enqueue(self, writer: WriterThread, frame: np.ndarray) -> None:
        """Enqueue a frame for this position on its assigned writer."""
        if self.write_state is None:  # pragma: no cover
            raise RuntimeError(f"Position {self.file_path!r} is not a TIFF file.")
        writer.enqueue(self.write_state, frame)

    def finalize(self, index_dims: tuple[Dimension, ...] | None) -> None:
        """Update metadata with the actual number of frames written.

        Parameters
        ----------
        index_dims : tuple[Dimension, ...] | None
            Dimensions used for storage indexing, or None if unavailable.
            (usually just T, C, Z)
        """
        # Update metadata based on actual frames written
        if self.write_state is None:
            # No thread means no TIFF file (e.g., companion OME-XML only)
            self.metadata_mirror.flush(force=True)
            return

        # No frames were routed to this position.  Leave it absent rather than
        # manufacturing an empty TIFF that contains no readable IFDs.
        if not self.write_state.started:
            return

        # Update dimension sizes and plane count based on actual frames written
        pixels = self.metadata_mirror.own_pixels()
        if self.write_state.frames_written and index_dims and pixels is not None:
            # Update the outermost dimension's size based on frames written
            # This handles both unbounded dims and incomplete bounded dims
            if index_dims:
                first, *inner = index_dims
                # Calculate actual size of outermost dimension
                inner_prod = math.prod([d.count or 1 for d in inner])
                actual_outer_size = max(
                    1, self.write_state.frames_written // inner_prod
                )
                setattr(pixels, f"size_{first.name.lower()}", actual_outer_size)

            # Update plane count
            if data_blocks := pixels.tiff_data_blocks:
                data_blocks[0].plane_count = self.write_state.frames_written
            self.metadata_mirror.flush(force=True)
        else:
            # Fallback: flush if any in-memory modifications have been made (e.g.
            # set_global_metadata on a 2D-only or zero-frame acquisition).  No force.
            self.metadata_mirror.flush()


class TiffBackend(ArrayBackend):
    """OME-TIFF backend using tifffile for sequential writes.

    TIFF files are written sequentially, with one file per position.
    The index parameter in write() is ignored since TIFF only supports
    sequential writing.
    """

    def __init__(self) -> None:
        self._finalized = False
        # Lock ordering: always acquire self._state_lock BEFORE any
        # PositionManager._lock. set_global_metadata() and finalize() follow
        # this order; inverting it in new code will deadlock.
        self._state_lock = threading.Lock()
        self._position_managers: dict[int, PositionManager] = {}
        self._storage_dims: tuple[Dimension, ...] | None = None
        self._dtype: str = ""
        self._frame_metadata: dict[int, list[dict[str, Any]]] = {}
        self._writer_threads: tuple[WriterThread, ...] = ()
        # position index -> (mirror holding its Image, index of that Image).
        # Not always the position's own file: a BinaryOnly stub has no Image to
        # hang a Plane off, so its per-frame metadata belongs in the master or
        # companion document instead.  See _build_plane_targets.
        self._plane_targets: dict[int, tuple[OmeXMLMirror, int]] = {}

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with TIFF backend."""
        # for now, assume we use the same compression strings as tifffile
        if settings.compression not in (None, "none") and not hasattr(
            tifffile.COMPRESSION, settings.compression.upper()
        ):  # pragma: no cover
            supported = {"none"} | set(tifffile.COMPRESSION.__members__.keys())
            return (
                f"Compression '{settings.compression}' is not supported by "
                f"TiffBackend. Supported: {supported}."
            )

        # Validate storage dimension names for OME-TIFF compatibility
        # OME-TIFF supports x, y, z, c, t (all StandardAxis except position)
        ome_dims = set("xyzct")
        for dim in settings.array_storage_dimensions:
            if dim.name.lower() not in ome_dims:  # pragma: no cover
                return (
                    f"Invalid dimension name '{dim.name}' for OME-TIFF. "
                    f"Valid names are: {', '.join(sorted(ome_dims))} "
                    f"(case-insensitive)."
                )

        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-TIFF files and writer threads."""

        self._finalized = False
        self._storage_dims = storage_dims = settings.array_storage_dimensions
        # Extract index keys, excluding Y and X, example: ['t', 'c', 'z']
        self._index_keys = [d.name for d in storage_dims[:-2]]
        self._dtype = settings.dtype
        self._frame_shape = tuple(d.count or 1 for d in self._storage_dims[-2:])

        # Extract and validate compression
        compression = None
        if settings.compression not in (None, "none"):
            compression = getattr(tifffile.COMPRESSION, settings.compression.upper())

        # Compute shape from storage dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)

        # Check if any dimension is unbounded
        has_unbounded = any(d.count is None for d in storage_dims)

        # Prepare OME-XML metadata mirrors
        # mapping of filepath -> OmeXMLMirror
        metas = prepare_metadata(settings)

        # Describe each position's writer state without opening its file.  Large
        # tiled acquisitions can contain hundreds of positions; eagerly opening all
        # of them creates hundreds of empty files and makes stream setup scale with
        # position count.
        expected_frames = None if has_unbounded else math.prod(shape[:-2] or (1,))
        for fname, meta_mirror in metas.items():
            write_state = None
            if meta_mirror.is_tiff:
                write_state = TiffWriteState(
                    file_path=fname,
                    dtype=self._dtype,
                    ome_xml=meta_mirror.model.to_xml(),
                    compression=compression,
                    expected_frames=expected_frames,
                )
            self._position_managers[meta_mirror.pos_idx] = PositionManager(
                file_path=fname,
                write_state=write_state,
                metadata_mirror=meta_mirror,
            )

        # A small fixed pool keeps thread usage bounded while allowing independent
        # position files to make progress concurrently on slow/network storage.
        # Starting the workers during prepare keeps thread startup out of acquisition
        # timing; TIFF files themselves remain lazy and open only on their first job.
        tiff_count = sum(
            manager.write_state is not None
            for manager in self._position_managers.values()
        )
        self._writer_threads = tuple(
            WriterThread(name=f"TiffWriterThread-{i}")
            for i in range(min(4, tiff_count))
        )
        for writer in self._writer_threads:
            writer.start()

        self._plane_targets = self._build_plane_targets()

    def _build_plane_targets(self) -> dict[int, tuple[OmeXMLMirror, int]]:
        """Map each position to the mirror and `Image` index that describe it.

        A `Plane` is a child of `Pixels`, so per-frame metadata can only be stored
        where that position's `Image` actually lives. In `redundant` and
        `self-contained` that is the position's own file, but a `master-tiff` stub
        or a `companion-file` TIFF holds no `Image` at all -- its metadata belongs
        in the one full-OME document, at the index matching its position.
        """
        full_mirrors = [
            mgr.metadata_mirror
            for mgr in self._position_managers.values()
            if mgr.metadata_mirror.model.binary_only is None
        ]
        targets: dict[int, tuple[OmeXMLMirror, int]] = {}
        for pos_idx, manager in self._position_managers.items():
            mirror = manager.metadata_mirror
            if mirror.model.binary_only is None:
                # describes itself; its own image_index already points at it
                targets[pos_idx] = (mirror, mirror.image_index)
            elif len(full_mirrors) == 1:
                # a stub: its Image lives in the single authoritative document,
                # where images are ordered by position
                targets[pos_idx] = (full_mirrors[0], pos_idx)
        return targets

    def _writer_for_position(self, position_index: int) -> WriterThread:
        """Return the stable pool worker assigned to ``position_index``."""
        if not self._writer_threads:  # pragma: no cover
            raise RuntimeError("TIFF writer pool is not initialized.")
        return self._writer_threads[position_index % len(self._writer_threads)]

    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
        *,
        frame_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write frame sequentially to the appropriate position's TIFF file.

        The index parameter is ignored since TIFF writes are sequential.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot write after finalize().")
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        manager = self._position_managers[position_index]
        manager.enqueue(self._writer_for_position(position_index), frame)

        # Accumulate frame metadata with storage index
        if frame_metadata is not None:
            self._append_frame_metadata(position_index, index, frame_metadata)

    def advance(self, indices: Sequence[tuple[int, tuple[int, ...]]]) -> None:
        """Write zero-filled placeholder frames to maintain sequential TIFF structure.

        TIFF files must be written sequentially. When frames are skipped during
        acquisition (e.g., autofocus failure), we write zero-filled placeholder
        frames to preserve the IFD order and structure.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot advance after finalize().")
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        if not indices:  # pragma: no cover
            return

        placeholder = np.zeros(self._frame_shape, dtype=self._dtype)
        # Write placeholder for each skipped frame
        for pos_idx, _storage_idx in indices:
            manager = self._position_managers[pos_idx]
            manager.enqueue(self._writer_for_position(pos_idx), placeholder)

    def _append_frame_metadata(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame_metadata: dict[str, Any],
    ) -> None:
        if position_index not in self._frame_metadata:
            self._frame_metadata[position_index] = []
        # self._frame_metadata[position_index].append(meta_with_idx)

        target = self._plane_targets.get(position_index)
        if target is None:  # pragma: no cover
            return
        mirror, image_index = target
        model = mirror.model
        images = model.images
        if 0 <= image_index < len(images):
            own_pixels = images[image_index].pixels
            # The annotation must land in the same document as the Plane that
            # references it, or the AnnotationRef dangles.
            if not (structured := model.structured_annotations):
                model.structured_annotations = structured = ome.StructuredAnnotations()
            map_annotations = structured.map_annotations
            # {"the_z": 0, "the_c": 1, ...}
            plane_kwargs = {
                f"the_{k}": v for k, v in zip(self._index_keys, index, strict=False)
            }
            plane_kwargs.update(
                {f"the_{k}": 0 for k in "tcz" if k not in self._index_keys}
            )

            extra_kwargs = {}
            for key, value in frame_metadata.items():
                if key in PLANE_KEYS:
                    plane_kwargs[key] = value
                else:
                    extra_kwargs[key] = value

            # meta_with_idx = {**frame_metadata, "storage_index": index}
            annotation_refs: list[ome.AnnotationRef] = []
            if extra_kwargs:
                annotation = ome.MapAnnotation(
                    value=ome.Map.model_validate(extra_kwargs)
                )
                map_annotations.append(annotation)
                annotation_refs.append(ome.AnnotationRef(id=annotation.id))
            own_pixels.planes.append(
                ome.Plane(**plane_kwargs, annotation_refs=annotation_refs)
            )
            mirror.mark_dirty()

    def _global_target_pos_idxs(self) -> list[int]:
        """Return pos_idxs of managers whose mirror holds the full OME model.

        Across every `multi_file_metadata` mode, the "full OME" mirrors are
        exactly the files that should carry global `MapAnnotation`s: the
        companion in companion-file mode, the master in master-tiff, the
        sole file in single-file, and every file in redundant and
        self-contained modes.
        `prepare_metadata` marks stub mirrors by setting
        ``model.binary_only``; full-OME mirrors leave it unset.
        """
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        return sorted(
            pos_idx
            for pos_idx, mgr in self._position_managers.items()
            if mgr.metadata_mirror.model.binary_only is None
        )

    def set_global_metadata(self, namespace: str, metadata: Mapping[str, Any]) -> None:
        """Attach acquisition-level metadata as a `MapAnnotation`.

        The annotation is placed under the target file's
        `OME.structured_annotations.map_annotations` with `Namespace=namespace` and is
        NOT referenced by any plane (distinguishing it from per-frame metadata). Same
        namespace replaces any prior value; different namespaces are siblings.

        Where the annotation lands on disk is driven by the OME-TIFF
        `multi_file_metadata` mode: single-file and `master-tiff` write to one file,
        `companion-file` writes to the companion, and `redundant` and
        `self-contained` write a copy into every per-position TIFF (matching those
        modes' "full OME in every file" guarantee).

        Can be called any time after `prepare()`, including after `close()`.
        Pre-close the update is deferred to `finalize()` (`tiffcomment` cannot
        safely run while a writer thread is still appending frames); post-close
        the update is flushed immediately via `tiffcomment` for TIFF mirrors
        or a direct file write for the companion file.
        """
        with self._state_lock:
            value = ome.Map.model_validate(
                {k: json.dumps(v) for k, v in metadata.items()}
            )

            for pos_idx in self._global_target_pos_idxs():
                manager = self._position_managers[pos_idx]
                with manager._lock:
                    manager.metadata_mirror.set_map_annotation(namespace, value)

                    # Post-finalize: writer threads are joined and files are
                    # closed, so flush directly.
                    if self._finalized:
                        manager.metadata_mirror.flush(force=True)

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        with self._state_lock:
            if self._finalized:
                return

            # Stop after all queued frames, then wait until every TIFF is closed.
            for writer in self._writer_threads:
                writer.stop()
            for writer in self._writer_threads:
                writer.join()

            # Finalize each position (wait for thread and update metadata)
            for manager in self._position_managers.values():
                index_dims = self._storage_dims[:-2] if self._storage_dims else None
                manager.finalize(index_dims)

            self._finalized = True

    def get_arrays(self) -> list[ArrayLike]:
        """Return array-like objects backed by TIFF files.

        If finalized and fully written: returns `FinalizedTiffArray` objects
        that read through *tifffile*'s page API (supports compression).

        Otherwise (live viewing, finalized-partial, or zero-frame): returns
        `LiveTiffArray` objects that read raw bytes at calculated offsets
        (requires uncompressed / contiguous writes).

        Returns
        -------
        list[ArrayLike]
            List of array-like objects (one per TIFF file).
        """

        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        arrays: list[ArrayLike] = []
        storage_dims = cast("tuple[Dimension]", self._storage_dims)
        for _, manager in sorted(self._position_managers.items()):
            if not manager.metadata_mirror.is_tiff:  # pragma: no cover
                continue  # Skip companion-only entries

            path = manager.file_path
            write_state = manager.write_state
            assert write_state is not None, f"No TIFF write state for {path}"

            if self._finalized:
                frames_written = write_state.frames_written
                shape = tuple(d.count or 1 for d in storage_dims)
                expected_frames = math.prod(shape[:-2] or (1,))
                if frames_written >= expected_frames and frames_written > 0:
                    # Fully written: read via tifffile pages (compression OK)
                    tf = tifffile.TiffFile(path)
                    arr = FinalizedTiffArray(tf, shape, self._dtype)
                    weakref.finalize(arr, tf.close)
                    arrays.append(arr)
                    continue

            # Everything else needs raw byte access (uncompressed)
            if write_state.compression is not None:
                raise NotImplementedError(
                    "Tiff viewing is not supported with compression enabled."
                )

            arrays.append(
                LiveTiffArray(
                    write_state=write_state,
                    file_path=path,
                    storage_dims=storage_dims,
                    dtype=self._dtype,
                )
            )

        return arrays

    def get_metadata(self) -> dict[int, ome.OME]:
        """Get the base OME metadata generated from acquisition settings.

        Returns a mapping of position indices to `ome_types.OME` objects.  The `OME`
        objects represent the metadata as it would appear in the TIFF or companion
        file for that position.

        !!! note
            The special "position index" of -1, if present, represents metadata
            in the companion.ome file, if applicable.

        Users can modify these objects as needed and pass a mapping of position indices
        to `ome_types.OME` objects back to `update_metadata()`.

        See the `ome-types` documentation for details on modifying OME metadata:
        <https://ome-types.readthedocs.io/en/latest/API/ome_types/>

        Returns
        -------
        dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects, or empty dict if
            prepare() has not been called yet.
        """
        if not self._position_managers:  # pragma: no cover
            return {}

        return {
            p_idx: manager.metadata_mirror.model.model_copy(deep=True)
            for p_idx, manager in self._position_managers.items()
        }

    def update_metadata(self, metadata: dict[int, ome.OME]) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be a dict mapping position indices to
        `ome_types.OME` instances, with the special index -1 representing the
        companion.ome file, if applicable.

        This method must be called AFTER exiting the stream context (after
        finalize() completes), as TIFF files must be closed before metadata
        can be updated.

        Parameters
        ----------
        metadata : dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects. Keys should match
            those returned by get_metadata().

        Raises
        ------
        TypeError
            If metadata is not a dict or values are not ome_types.model.OME instances.
        KeyError
            If a position index in metadata doesn't correspond to a position.
        RuntimeError
            If called before finalize() completes, or if metadata update fails.
        """
        if not self._finalized:  # pragma: no cover
            raise RuntimeError(
                "update_metadata() must be called after the stream context exits. "
                "TIFF files must be closed before metadata can be updated."
            )

        if not isinstance(metadata, dict):
            raise TypeError(
                "Expected dict[int, ome_types.model.OME] metadata, "
                f"got {type(metadata)}"
            )

        for pos_idx, meta in metadata.items():
            if not isinstance(meta, ome.OME):
                raise TypeError(
                    f"Expected ome_types.model.OME for position {pos_idx}, "
                    f"got {type(meta)}"
                )

            try:
                # not calling deep copy here, since this is currently only ever called
                # after finalize().  i.e. we're done.
                self._position_managers[pos_idx].update_metadata(meta, flush=True)
            except KeyError as e:  # pragma: no cover
                raise KeyError(f"Unknown position index: {pos_idx}") from e


class TiffWriteState:
    """Per-position state shared by a writer worker and live readers."""

    def __init__(
        self,
        file_path: str,
        dtype: str,
        ome_xml: str = "",
        pixelsize: float = 1.0,
        compression: tifffile.COMPRESSION | None = None,
        expected_frames: int | None = None,
    ) -> None:
        self.file_path = file_path
        self.dtype = dtype
        # Passing bytes preserves non-ASCII OME units such as 'µ'.  Serialize
        # during store preparation so metadata work never delays acquisition.
        self.ome_xml_bytes = ome_xml.encode("utf-8")
        self.resolution = 1 / pixelsize
        self.compression = compression
        self.expected_frames = expected_frames
        self.frames_enqueued = 0
        self.frames_written = 0  # Track actual frames written for unbounded dims
        self.state_lock = threading.Lock()  # Synchronize with readers
        self.data_offset: int | None = None  # Byte offset where frame data starts
        self.complete = False
        self.failed = False

    @property
    def started(self) -> bool:
        """Whether this position has received at least one frame."""
        with self.state_lock:
            return self.frames_enqueued > 0


class WriterThread(threading.Thread):
    """One worker in the bounded pool of sequential TIFF writers."""

    def __init__(self, name: str) -> None:
        super().__init__(daemon=True, name=name)
        self._image_queue: Queue[tuple[TiffWriteState, np.ndarray] | None] = Queue()
        self._writers: dict[str, tifffile.TiffWriter] = {}

    def enqueue(self, state: TiffWriteState, frame: np.ndarray) -> None:
        """Enqueue ``frame`` without doing filesystem work on the caller thread."""
        with state.state_lock:
            state.frames_enqueued += 1
        self._image_queue.put((state, frame))

    def stop(self) -> None:
        """Stop after all previously queued frames have been written."""
        self._image_queue.put(None)

    def run(self) -> None:
        """Write frames from the assigned position queues in arrival order."""
        try:
            while (job := self._image_queue.get()) is not None:
                state, frame = job
                try:
                    self._write_frame(state, frame)
                except Exception as e:  # pragma: no cover
                    self._close_writer(state.file_path)
                    state.failed = True
                    warnings.warn(
                        f"Unexpected error writing {state.file_path!r}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            for file_path in tuple(self._writers):
                self._close_writer(file_path)

    def _write_frame(self, state: TiffWriteState, frame: np.ndarray) -> None:
        if state.failed:
            return
        if state.complete:  # pragma: no cover
            raise RuntimeError("Received a frame after the position was complete.")

        writer = self._writers.get(state.file_path)
        if writer is None:
            # File creation may be slow on network storage.  It happens only here,
            # never in append() or on the acquisition thread.
            writer = tifffile.TiffWriter(
                state.file_path, bigtiff=True, ome=False, shaped=False
            )
            self._writers[state.file_path] = writer

        frame_index = state.frames_written
        writer.write(
            frame,
            contiguous=state.compression is None,
            dtype=state.dtype,
            resolution=(state.resolution, state.resolution),
            resolutionunit=tifffile.RESUNIT.MICROMETER,
            photometric=tifffile.PHOTOMETRIC.MINISBLACK,
            description=state.ome_xml_bytes if frame_index == 0 else None,
            compression=state.compression,
        )

        with state.state_lock:
            if frame_index == 0 and state.data_offset is None:
                try:
                    # ! private attribute access - relies on tifffile internals
                    state.data_offset = writer._dataoffset
                except AttributeError:  # pragma: no cover
                    raise RuntimeError(
                        "tifffile.TiffWriter has no _dataoffset attribute. "
                        "Cannot determine frame data offset for live viewing. "
                        "Please report this issue with the version of tifffile "
                        "you're using."
                    ) from None
            state.frames_written += 1
            complete = state.frames_written == state.expected_frames
            state.complete = complete

        if complete:
            self._close_writer(state.file_path)

    def _close_writer(self, file_path: str) -> None:
        if writer := self._writers.pop(file_path, None):
            with suppress(Exception):
                writer.close()
