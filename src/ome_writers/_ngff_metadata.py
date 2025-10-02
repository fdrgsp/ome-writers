from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from yaozarrs.v05 import Plate

    from ome_writers._dimensions import Dimension

from ome_writers.model import dims_to_ngff_axes


def ngff_meta_v5(
    array_dims: Mapping[str, Sequence[Dimension]],
    plate: Plate | None = None,
) -> dict:
    """Create OME NGFF v0.5 metadata.

    Parameters
    ----------
    array_dims : Mapping[str, Sequence[DimensionInfo]]
        A mapping of array paths to their corresponding dimension information.
        Each key is the path to a zarr array, and the value is a sequence of
        DimensionInfo objects describing the dimensions of that array.
    plate : Plate | None, optional
        HCS plate metadata. If provided, plate metadata will be included
        in the OME metadata following the OME-NGFF 0.5 specification.

    Example
    -------
    >>> from ome_writers import DimensionInfo, ome_meta_v5
    >>> array_dims = {
        "0": [
            DimensionInfo(label="t", size=1, unit=(1.0, "s")),
            DimensionInfo(label="c", size=1, unit=(1.0, "s")),
            DimensionInfo(label="z", size=1, unit=(1.0, "s")),
            DimensionInfo(label="y", size=1, unit=(1.0, "s")),
            DimensionInfo(label="x", size=1, unit=(1.0, "s")),
        ],
    }
    >>> ome_meta = ome_meta_v5(array_dims)
    """
    # Group arrays by their axes to create multiscales entries
    multiscales: dict[str, dict] = {}

    for array_path, dims in array_dims.items():
        axes, scales = dims_to_ngff_axes(dims)
        ct = {"scale": scales, "type": "scale"}
        ds = {"path": array_path, "coordinateTransformations": [ct]}

        # Create a hashable key from axes for grouping
        axes_key = str(axes)
        # Create a new entry for this axes configuration if it doesn't exist
        # (in the case where multiple arrays share the same axes, we want to
        # create multiple datasets under the same multiscale entry, rather than
        # creating a new multiscale entry with a single dataset each time)
        multiscale = multiscales.setdefault(axes_key, {"axes": axes, "datasets": []})

        # Add the dataset to the corresponding group
        multiscale["datasets"].append(ds)

    # Build the base OME metadata
    ome_attrs: dict[str, Any] = {
        "version": "0.5",
        "multiscales": list(multiscales.values()),
    }

    # Add HCS-specific metadata if provided
    if plate is not None:
        ome_attrs["plate"] = _plate_to_dict(plate)

    attrs = {"ome": ome_attrs}
    return attrs


def _plate_to_dict(plate: Plate) -> dict:
    """Convert a Plate object to OME-NGFF 0.5 compliant dictionary."""
    plate_dict = {
        "columns": [{"name": col.name} for col in plate.columns],
        "rows": [{"name": row.name} for row in plate.rows],
        "wells": [
            {
                "path": well.path,
                "rowIndex": well.rowIndex,
                "columnIndex": well.columnIndex,
            }
            for well in plate.wells
        ],
        "version": plate.version,
    }

    if plate.acquisitions is not None:
        plate_dict["acquisitions"] = [
            {
                "id": acq.id,
                **({} if acq.name is None else {"name": acq.name}),
                **(
                    {}
                    if acq.maximumfieldcount is None
                    else {"maximumfieldcount": acq.maximumfieldcount}
                ),
                **({} if acq.description is None else {"description": acq.description}),
                **({} if acq.starttime is None else {"starttime": acq.starttime}),
                **({} if acq.endtime is None else {"endtime": acq.endtime}),
            }
            for acq in plate.acquisitions
        ]

    if plate.field_count is not None:
        plate_dict["field_count"] = plate.field_count

    if plate.name is not None:
        plate_dict["name"] = plate.name

    return plate_dict
