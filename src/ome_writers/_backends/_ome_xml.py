from __future__ import annotations

import itertools
import uuid
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import ome_types
import ome_types.model as ome
import tifffile

from ome_writers._schema import OmeTiffFormat
from ome_writers._units import ngff_to_ome_unit
from ome_writers._util import spatial_role_indices

if TYPE_CHECKING:
    from ome_writers._schema import AcquisitionSettings, Channel, Dimension, Position

BinaryOnly = ome.OME.BinaryOnly
COMPANION_IDX = -1  # special index for companion.ome files


class MultiFileMetadata(Enum):
    """Multi-file metadata arrangement modes.

    These modes control how OME-XML metadata is distributed across
    multiple TIFF files in multi-position acquisitions.
    """

    REDUNDANT = "redundant"
    """Each TIFF file has complete OME-XML metadata."""
    MASTER_TIFF = "master-tiff"
    """First TIFF has full OME-XML, others have BinData references."""
    COMPANION = "companion-file"
    """All TIFFs have BinData only, full OME-XML in separate companion file."""
    SELF_CONTAINED = "self-contained"
    """Each TIFF describes only its own position, with no references to siblings.

    Unlike the other three modes, this does not produce a multi-file OME-TIFF
    *set*: it produces N independent single-series OME-TIFFs that happen to share
    a directory.  Each file can be opened, moved, or deleted on its own.
    """


class OmeXMLMirror:
    """In-memory mirror of an on-disk OME-XML structure, with dirty tracking.

    Parameters
    ----------
    path : str
        Complete path to the OME-XML file or TIFF file that will contain the OME-XML
        metadata.
    model : ome.OME | None, optional
        Initial OME model to use. If None, an empty OME model is created.
    image_index : int | None, optional
        Index of *this position's* `Image` within `model.images`. Full-OME mirrors
        carry every position, so the image sits at its position index (the default).
        Self-contained mirrors carry only their own image, at index 0.
    """

    def __init__(
        self,
        path: str | Path,
        pos_idx: int | None,
        model: ome.OME | None = None,
        image_index: int | None = None,
    ) -> None:
        self.path: str = str(path)
        self.pos_idx: int = COMPANION_IDX if pos_idx is None else pos_idx
        self.model = model or ome.OME()
        self.image_index: int = self.pos_idx if image_index is None else image_index
        self._dirty: bool = False

    def own_pixels(self) -> ome.Pixels | None:
        """Return the `Pixels` of this position's own `Image`, if present.

        Returns None for BinaryOnly stubs and the companion mirror, which hold no
        image of their own.
        """
        images = self.model.images
        if 0 <= self.image_index < len(images):
            return images[self.image_index].pixels
        return None

    def mark_dirty(self) -> None:
        """Mark the OME-XML as dirty (modified)."""
        self._dirty = True

    def set_map_annotation(self, namespace: str, value: ome.Map) -> None:
        """Set a `MapAnnotation` keyed by `namespace`, preserving order.

        If a `MapAnnotation` with the given `namespace` already exists in
        `structured_annotations.map_annotations`, it is replaced *in place*
        so sibling annotations keep their relative order. Otherwise a new
        entry is appended. Any later duplicates with the same namespace are
        dropped as a defensive measure. The mirror is marked dirty.

        This method is not internally locked; callers must hold whatever
        lock protects concurrent access to this mirror (e.g. the owning
        `PositionManager._lock` in `TiffBackend`).
        """
        if not (structured := self.model.structured_annotations):
            self.model.structured_annotations = structured = ome.StructuredAnnotations()

        new_ann = ome.MapAnnotation(namespace=namespace, value=value)
        existing = structured.map_annotations
        idx = next(
            (i for i, a in enumerate(existing) if a.namespace == namespace),
            None,
        )
        if idx is None:
            existing.append(new_ann)
        else:
            existing[idx] = new_ann
            if any(a.namespace == namespace for a in existing[idx + 1 :]):
                structured.map_annotations = [
                    a
                    for i, a in enumerate(existing)
                    if i <= idx or a.namespace != namespace
                ]
        self._dirty = True

    @property
    def is_tiff(self) -> bool:
        """Whether the OME-XML is stored in a TIFF file."""
        return self.path.lower().endswith((".tiff", ".tif"))

    def flush(self, *, force: bool = False) -> None:
        """Mark the OME-XML as clean (not modified)."""
        if not self._dirty and not force:  # pragma: no cover
            return

        xml_bytes = self.model.to_xml().encode("utf-8")
        if self.is_tiff:
            try:
                tifffile.tiffcomment(self.path, comment=xml_bytes)
            except FileNotFoundError:  # pragma: no cover
                warnings.warn(
                    f"TIFF file {self.path} not found when writing OME-XML comment.",
                    stacklevel=2,
                )
        else:
            # companion file
            with open(self.path, mode="wb") as f:
                f.write(xml_bytes)

        self._dirty = False


def prepare_metadata(settings: AcquisitionSettings) -> dict[str, OmeXMLMirror]:
    """Create OME-XML mirrors based on the acquisition settings.

    Parameters
    ----------
    settings : AcquisitionSettings
        The acquisition settings containing file paths and format configuration.

    Returns
    -------
    dict[str, OmeXMLMirror]
        Mapping of file paths to their OME-XML mirrors.

    Notes
    -----
    Invariant relied on by `TiffBackend._global_target_pos_idxs`: stub
    mirrors (BinData-only references to another file) MUST set
    ``model.binary_only``, and full-OME mirrors MUST leave it unset. The
    backend uses that flag to decide which file(s) receive global
    `MapAnnotation` updates.  Self-contained mirrors are full-OME by this
    definition, so (as in redundant mode) every file receives a copy.
    """
    if not isinstance(settings.format, OmeTiffFormat):
        raise ValueError("Expected settings.format to be an OmeTiffFormat instance.")

    if any(
        dim.name.lower() not in "tczyx"
        for dim in settings.dimensions
        if dim.type != "position"
    ):  # pragma: no cover
        raise ValueError("Dimension names must be one of 't', 'c', 'z', 'y', 'x'")

    # Determine file structure: single-file vs multi-file
    prefer = settings.format.prefer_single_file
    num_pos = len(settings.positions)
    single_file = prefer == "always" or (prefer == "auto" and num_pos <= 1)

    if single_file and num_pos > 1:
        raise NotImplementedError(
            "Single-file structure with multiple positions is not yet "
            "supported. Use prefer_single_file='auto' or 'never' for "
            "multi-position acquisitions."
        )

    # For multi-file mode, determine metadata arrangement
    metadata_mode: MultiFileMetadata | None = None
    if not single_file:
        # Enum values now match field values, so direct cast works
        metadata_mode = MultiFileMetadata(settings.format.multi_file_metadata)

    # Generate file info (path + UUID) for each position
    file_infos = _generate_file_infos(settings, single_file)
    for info in file_infos:
        # Handle overwrite
        path = Path(info.path)
        if path.exists():
            if not settings.overwrite:
                raise FileExistsError(
                    f"File {path} already exists. Use overwrite=True to overwrite it."
                )
            path.unlink()

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

    # Create mirrors based on file structure and metadata arrangement
    mirrors: dict[str, OmeXMLMirror] = {}

    if metadata_mode == MultiFileMetadata.SELF_CONTAINED:
        # Each file gets its own single-image model.  Note that no full model is
        # built: the other modes copy an N-image model into N files, which makes
        # store preparation quadratic in position count.
        ctx = _image_context(settings)
        for info, pos in zip(file_infos, settings.positions, strict=True):
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=_build_self_contained_model(ctx, settings, info, pos),
                image_index=0,
            )
        return mirrors

    # Build the complete OME model with all series
    full_model = _build_full_model(settings, file_infos, single_file)

    if single_file:
        # Single file contains everything
        info = file_infos[0]
        mirrors[info.path] = OmeXMLMirror(
            path=info.path, pos_idx=info.pos_idx, model=full_model
        )

    elif metadata_mode == MultiFileMetadata.REDUNDANT:
        # Each file gets a full copy, differing only in root UUID
        for info in file_infos:
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=full_model.model_copy(deep=True, update={"uuid": info.uuid}),
            )

    elif metadata_mode == MultiFileMetadata.MASTER_TIFF:
        # First file is master with full metadata
        master_info = file_infos[0]
        full_model.uuid = master_info.uuid
        mirrors[master_info.path] = OmeXMLMirror(
            path=master_info.path,
            pos_idx=master_info.pos_idx,
            model=full_model,
        )

        # Other files get BinaryOnly
        for info in file_infos[1:]:
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=ome_types.OME(
                    uuid=info.uuid,
                    binary_only=BinaryOnly(
                        metadata_file=master_info.path,
                        uuid=master_info.uuid,
                    ),
                ),
            )

    elif metadata_mode == MultiFileMetadata.COMPANION:
        # Companion file goes in the output directory alongside TIFF files
        output_path = Path(settings.output_path)
        companion_filename = settings.format.companion_file
        companion_path = str(output_path / companion_filename)
        companion_uuid = _make_uuid()

        # Companion file gets full metadata
        full_model.uuid = companion_uuid
        mirrors[companion_path] = OmeXMLMirror(
            path=companion_path,
            pos_idx=COMPANION_IDX,
            model=full_model,
        )

        # All TIFF files get BinaryOnly
        for info in file_infos:
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=ome_types.OME(
                    uuid=info.uuid,
                    binary_only=BinaryOnly(
                        metadata_file=companion_path,
                        uuid=companion_uuid,
                    ),
                ),
            )

    return mirrors


# ---------------------------------------------------------------------------
# Helper types and functions
# ---------------------------------------------------------------------------


class FileInfo(NamedTuple):
    """Simple container for file path and UUID."""

    path: str
    uuid: str
    pos_idx: int | None  # None for companion file


def _make_uuid() -> str:
    """Generate a URN-formatted UUID."""
    return f"urn:uuid:{uuid.uuid4()}"


def _stage_identity(pos: Position, flat_index: int) -> tuple[object, ...]:
    """Return the key identifying the *stage location* a position belongs to.

    A flattened position dimension interleaves two different things: the
    stage location visited, and which tile of that location the frame is.
    This key names only the former, so that repeats of it are recognizable
    as tiles of one location.

    Only a position carrying grid coordinates is a tile of something. Without
    them the position is a location in its own right -- an ordinary stage
    position, or a randomly sampled point within a well (`RandomPoints`),
    which is a distinct position rather than a tile of a regular grid -- so
    it gets an identity unique to itself.

    For a gridded well plate the location is the **well**: `Position.name`
    there identifies the field of view within the well (e.g. "fov0"), so it
    is deliberately excluded. Otherwise the name identifies the stage
    position itself and is shared by all of its grid tiles.
    """
    if pos.grid_row is None or pos.grid_column is None:
        return ("", flat_index)
    if pos.plate_row is not None or pos.plate_column is not None:
        return (pos.plate_row, pos.plate_column)
    return (pos.name,)


def _position_filename_suffixes(positions: list[Position]) -> list[str]:
    """Return a per-position `[_{well}]_p###[_r###_c###]` filename suffix.

    `p` is the position's *stage* ordinal (first-appearance order among
    distinct stage identities, see `_stage_identity`), not its flat index
    in `positions` -- for a single-tile-per-location acquisition the two
    coincide, but a grid maps several flat entries onto one stage location,
    and the flat index alone can't tell those apart (e.g. two stage
    positions each with their own 2-tile grid previously all fell under the
    same misleading `p000..p003` range, reordering with `axis_order`).

    Plate positions are additionally prefixed with their well (e.g. `_A1`),
    which is what actually identifies the location to a human reader.

    A `_r{row:03}_c{col:03}` tile suffix is appended only when that stage
    identity actually repeats -- i.e. the location really does have more than
    one tile -- so a single tile per location (including a grid-only
    acquisition, which enumerates its tiles as the positions themselves)
    keeps plain `_p###` naming rather than a redundant `p001_r000_c001`.
    """
    keys = [_stage_identity(pos, i) for i, pos in enumerate(positions)]

    identity_counts: dict[tuple[object, ...], int] = {}
    for key in keys:
        identity_counts[key] = identity_counts.get(key, 0) + 1

    stage_ordinal: dict[tuple[object, ...], int] = {}
    suffixes: list[str] = []
    for pos, key in zip(positions, keys, strict=True):
        if key not in stage_ordinal:
            stage_ordinal[key] = len(stage_ordinal)

        suffix = ""
        if pos.plate_row is not None and pos.plate_column is not None:
            suffix += f"_{pos.plate_row}{pos.plate_column}"
        suffix += f"_p{stage_ordinal[key]:03}"
        if identity_counts[key] > 1:
            # Only a grid-coordinate-bearing position can share an identity,
            # so row/column are known to be set here.
            suffix += f"_r{pos.grid_row:03}_c{pos.grid_column:03}"
        suffixes.append(suffix)

    # Distinct files must never collapse onto one path: prepare_metadata keys
    # its mirrors by path, so a duplicate silently drops a position's writer.
    # Nothing above should produce one, but a malformed grid (two tiles
    # reporting the same row/column) would, so break any tie explicitly.
    if len(set(suffixes)) != len(suffixes):  # pragma: no cover
        seen: set[str] = set()
        for i, suffix in enumerate(suffixes):
            if suffix in seen:
                suffixes[i] = f"{suffix}_{i:03}"
            seen.add(suffixes[i])
    return suffixes


def _generate_file_infos(
    settings: AcquisitionSettings, single_file: bool
) -> list[FileInfo]:
    """Generate file paths and UUIDs for each position."""

    output_path = Path(settings.output_path).expanduser().resolve()
    positions = settings.positions

    if single_file:
        # Single file for all positions
        return [FileInfo(path=str(output_path), uuid=_make_uuid(), pos_idx=0)]

    # XXX: if we switch to using pos.name directly, we need to sanitize it
    # def _safename(name: str) -> str:
    #     """Sanitize a string to be safe for filenames."""
    #     if str.isdigit(name):
    #         return f"p{name:04}"
    #     return name.replace("/", "_").replace("\\", "_")

    # Multi-file: one file per position inside output_path directory
    # output_path is now a directory for multi-file modes
    output_dir = output_path
    stem = output_dir.name
    extension = settings.format.suffix

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = _position_filename_suffixes(list(positions))
    return [
        FileInfo(
            path=str(output_dir / f"{stem}{suffix}{extension}"),
            uuid=_make_uuid(),
            pos_idx=idx,
        )
        for idx, suffix in enumerate(suffixes)
    ]


class _ImageContext(NamedTuple):
    """Position-independent inputs for building `ome.Image` elements.

    Computed once per acquisition so that building N per-position models costs
    O(N) rather than re-deriving shared structure for every position.
    """

    dims: list[Dimension]
    dimension_order: str
    channel_dim: Dimension | None
    size_x: int
    size_y: int
    size_z: int
    size_c: int
    size_t: int
    physical_sizes: dict
    dtype: str

    @property
    def planes_per_series(self) -> int:
        return self.size_t * self.size_c * self.size_z


def _image_context(settings: AcquisitionSettings) -> _ImageContext:
    """Derive the shared, position-independent `ome.Image` inputs."""
    dims = [d for d in settings.dimensions if d.type != "position"]
    pixel_sizes = {"z": 1, "c": 1, "t": 1}
    pixel_sizes.update({d.name.lower(): d.count or 1 for d in dims})
    return _ImageContext(
        dims=dims,
        dimension_order=_get_dimension_order(dims),
        channel_dim=next(
            (d for d in dims if d.type == "channel" or d.name.lower() == "c"), None
        ),
        size_x=pixel_sizes["x"],
        size_y=pixel_sizes["y"],
        size_z=pixel_sizes["z"],
        size_c=pixel_sizes["c"],
        size_t=pixel_sizes["t"],
        physical_sizes=_get_physical_sizes(dims),
        dtype=settings.dtype,
    )


def _build_image(
    ctx: _ImageContext, pos: Position, tiff_data: ome.TiffData, idx: int
) -> ome.Image:
    """Build one `ome.Image`, numbering its OME IDs with `idx`.

    `idx` is the index *within the containing document*, which is the position
    index for full-OME models but always 0 for self-contained ones.
    """
    if ctx.channel_dim and ctx.channel_dim.coords:
        channels = [
            _cast_channel(omw_channel=c, id=f"Channel:{idx}:{cidx}")
            for cidx, c in enumerate(ctx.channel_dim.coords)
        ]
    else:
        channels = [ome.Channel(id=f"Channel:{idx}:{c}") for c in range(ctx.size_c)]

    pixels = ome.Pixels(
        id=f"Pixels:{idx}",
        dimension_order=ctx.dimension_order,
        size_x=ctx.size_x,
        size_y=ctx.size_y,
        size_z=ctx.size_z,
        size_c=ctx.size_c,
        size_t=ctx.size_t,
        **ctx.physical_sizes,
        type=ctx.dtype,
        # big_endian=False,
        channels=channels,
        tiff_data_blocks=[tiff_data],
    )
    return ome.Image(
        id=f"Image:{idx}",
        name=pos.name,
        pixels=pixels,
        acquisition_date=datetime.now(timezone.utc),
        stage_label=_build_stage_label(pos, ctx.dims),
    )


def _build_full_model(
    settings: AcquisitionSettings, file_infos: list[FileInfo], single_file: bool
) -> ome_types.OME:
    """Build complete OME model with all series/images."""
    ctx = _image_context(settings)
    planes_per_series = ctx.planes_per_series

    # Track cumulative IFD offset for single-file mode
    ifd_offset = 0

    images: list[ome.Image] = []
    for i, pos in enumerate(settings.positions):
        # For single file, all series reference the same file (no UUID children)
        # For multi-file, each series references all files via UUID children
        if single_file:
            tiff_data = ome.TiffData(ifd=ifd_offset, plane_count=planes_per_series)
            ifd_offset += planes_per_series
        else:
            # Create TiffData for this series pointing to its file
            # Each series lives in one file, so one TiffData with UUID
            file_info = file_infos[i]
            relative_path = Path(file_info.path).name
            tiff_data = ome.TiffData(
                plane_count=planes_per_series,
                uuid=ome.TiffData.UUID(file_name=relative_path, value=file_info.uuid),
            )

        images.append(_build_image(ctx, pos, tiff_data, i))

    plates = _build_plates(settings)
    return ome_types.OME(uuid=_make_uuid(), images=images, plates=plates)


def _build_self_contained_model(
    ctx: _ImageContext, settings: AcquisitionSettings, info: FileInfo, pos: Position
) -> ome_types.OME:
    """Build a standalone single-series OME model for one position's file.

    The `TiffData` carries no `UUID` child: per the OME schema that element "must be
    used when the IFDs are located in another file", and omitting it means the IFDs
    live in the file the OME-XML was read from.  That is what makes the file
    independently readable.
    """
    tiff_data = ome.TiffData(ifd=0, plane_count=ctx.planes_per_series)
    image = _build_image(ctx, pos, tiff_data, 0)
    plates = _build_plates(settings, only_pos_idx=info.pos_idx)
    return ome_types.OME(uuid=info.uuid, images=[image], plates=plates)


VALID_ORDERS = [x.value for x in ome.Pixels_DimensionOrder]


def _get_dimension_order(dims: list[Dimension]) -> str:
    # suffix is some combination of ZCT in order of appearance
    # reversed to match OME dimension order conventions (f)
    suffix = "".join(d.name.upper() for d in dims if d.name.upper() not in "XY")[::-1]
    for order in VALID_ORDERS:
        if order[2:].startswith(suffix):
            return order
    raise ValueError(f"No valid order matches {suffix}")  # pragma: no cover


def _build_stage_label(pos: Position, dims: list[Dimension]) -> ome.StageLabel | None:
    """Build `ome.StageLabel` from a `Position` if any XYZ coord is present.

    Coordinates are written verbatim (unlike the NGFF translation, no
    `Dimension.translation` is added: `StageLabel` is the stage position of the
    image, not the coordinate of its first element). Units are derived from the
    spatial dim that plays the corresponding role (per `spatial_role_indices`),
    converted to an OME-XML `UnitsLength` string via `ngff_to_ome_unit`. Axes with
    no coord are left unset and take `StageLabel`'s default (`REFERENCEFRAME`).
    Returns `None` when no coord is available so no empty `StageLabel` is attached.
    """
    if pos.x_coord is None and pos.y_coord is None and pos.z_coord is None:
        return None

    roles = spatial_role_indices(dims)
    kwargs: dict[str, object] = {"name": pos.name}
    for axis, value in (("x", pos.x_coord), ("y", pos.y_coord), ("z", pos.z_coord)):
        if value is not None:
            kwargs[axis] = value
            idx = roles.get(axis)
            if idx is not None and (unit := dims[idx].unit):
                if ome_unit := ngff_to_ome_unit(unit):
                    kwargs[f"{axis}_unit"] = ome_unit
    return ome.StageLabel(**kwargs)


def _get_physical_sizes(dims: list[Dimension]) -> dict:
    output = {}
    dims_map = {dim.name.lower(): dim for dim in dims}
    for axis in ["x", "y", "z"]:
        if (dim := dims_map.get(axis)) and dim.scale is not None:
            output[f"physical_size_{axis}"] = dim.scale
            # if dim.type is space, it's guaranteed to be a valid NGFF unit
            if (
                dim.type == "space"
                and dim.unit
                and (ome_unit := ngff_to_ome_unit(dim.unit))
            ):
                output[f"physical_size_{axis}_unit"] = ome_unit
    return output


def _build_plates(
    settings: AcquisitionSettings, only_pos_idx: int | None = None
) -> list[ome.Plate]:
    """Build OME Plate with Wells and WellSamples linking to Images.

    Each position maps to a WellSample, which links to an Image via ImageRef.
    Wells are determined by unique (plate_row, plate_column) combinations.

    When `only_pos_idx` is given, the plate is reduced to just the well and field
    that this one position occupies, and its `ImageRef` points at `Image:0`.  The
    `Plate` still reports the full row/column count, so a reader can tell which
    well of which plate layout the file came from, but no `ImageRef` points at an
    image living in another file.  `WellSample` keeps its original index so it
    stays traceable to its place in the acquisition.
    """
    if not (positions := settings.positions) or not (plate := settings.plate):
        return []

    # all known (row, column) keys based on plate definition
    valid_keys = set(itertools.product(plate.row_names, plate.column_names))

    # Group positions by well (row, column), filtering out invalid coordinates
    wells_map: dict[tuple[str, str], list[tuple[int, Position]]] = {}
    for idx, pos in enumerate(positions):
        if only_pos_idx is not None and idx != only_pos_idx:
            continue
        # in AcquisitionSettings._validate_plate_positions, we already warned the user
        # that positions with with plate_row/plate_column that aren't represented
        # in the plate definition will be skipped from the metadata.
        # So here we just skip them here silently.
        if pos.plate_row is None or pos.plate_column is None:
            continue
        key = (pos.plate_row, pos.plate_column)
        if key in valid_keys:
            wells_map.setdefault(key, []).append((idx, pos))

    if only_pos_idx is not None and not wells_map:
        # This position is not on the plate; emit no Plate rather than an empty one.
        return []

    # Build Well objects with WellSamples
    wells: list[ome.Well] = []
    for (row_name, col_name), positions in wells_map.items():
        row_idx = plate.row_names.index(row_name)
        col_idx = plate.column_names.index(col_name)

        well_samples = [
            ome.WellSample(
                id=f"WellSample:{idx}",
                index=idx,
                image_ref=ome.ImageRef(
                    id="Image:0" if only_pos_idx is not None else f"Image:{idx}"
                ),
            )
            for idx, _pos in positions
        ]

        wells.append(
            ome.Well(
                id=f"Well:{row_idx}_{col_idx}",
                row=row_idx,
                column=col_idx,
                well_samples=well_samples,
            )
        )

    row_conv = _infer_naming_convention(plate.row_names)
    col_conv = _infer_naming_convention(plate.column_names)

    return [
        ome.Plate(
            id="Plate:0",
            name=plate.name,
            rows=len(plate.row_names),
            columns=len(plate.column_names),
            row_naming_convention=row_conv,
            column_naming_convention=col_conv,
            wells=wells,
        )
    ]


def _infer_naming_convention(names: list[str]) -> ome.NamingConvention | None:
    """Infer naming convention from a list of names."""
    if all(n.isalpha() for n in names):
        return ome.NamingConvention.LETTER
    if all(n.isdigit() for n in names):
        return ome.NamingConvention.NUMBER
    return None  # pragma: no cover


def _cast_channel(omw_channel: Channel, id: str) -> ome.Channel:
    """Cast an ome_writers Channel to an ome_types.model Channel."""
    color = {"color": omw_channel.color.as_rgb_tuple()} if omw_channel.color else {}
    return ome.Channel(
        id=id,
        name=omw_channel.name,
        **color,
        emission_wavelength=omw_channel.emission_wavelength_nm,
        emission_wavelength_unit=ome.UnitsLength.NANOMETER,
        excitation_wavelength=omw_channel.excitation_wavelength_nm,
        excitation_wavelength_unit=ome.UnitsLength.NANOMETER,
        fluor=omw_channel.fluorophore,
    )
