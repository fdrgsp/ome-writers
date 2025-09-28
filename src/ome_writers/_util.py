from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast, get_args

import numpy as np
import numpy.typing as npt

from ome_writers.model import (
    ColumnNGFF,
    Dimension,
    DimensionLabel,
    PlateNGFF,
    RowNGFF,
    WellImageNGFF,
    WellInPlateNGFF,
    WellNGFF,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import useq

    from ome_writers import UnitTuple


VALID_LABELS = get_args(DimensionLabel)
DEFAULT_UNITS: Mapping[DimensionLabel, UnitTuple | None] = {
    "t": (1.0, "s"),
    "z": (1.0, "um"),
    "y": (1.0, "um"),
    "x": (1.0, "um"),
    "c": None,
    "p": None,
}


def fake_data_for_sizes(
    sizes: Mapping[str, int],
    *,
    dtype: npt.DTypeLike = np.uint16,
    chunk_sizes: Mapping[str, int] | None = None,
) -> tuple[Iterator[np.ndarray], list[Dimension], np.dtype]:
    """Simple helper function to create a data generator and dimensions.

    Provide the sizes of the dimensions you would like to "acquire", along with the
    datatype and chunk sizes. The function will return a generator that yields
    2-D (YX) planes of data, along with the dimension information and the dtype.

    This can be passed to create_stream to create a stream for writing data.

    Parameters
    ----------
    sizes : Mapping[str, int]
        A mapping of dimension labels to their sizes. Must include 'y' and 'x'.
    dtype : np.typing.DTypeLike, optional
        The data type of the generated data. Defaults to np.uint16.
    chunk_sizes : Mapping[str, int] | None, optional
        A mapping of dimension labels to their chunk sizes. If None, defaults to 1 for
        all dimensions, besizes 'y' and 'x', which default to their full sizes.
    """
    if not {"y", "x"} <= sizes.keys():  # pragma: no cover
        raise ValueError("sizes must include both 'y' and 'x'")
    if not all(k in VALID_LABELS for k in sizes):  # pragma: no cover
        raise ValueError(
            f"Invalid dimension labels in sizes: {sizes.keys() - set(VALID_LABELS)}"
        )

    _chunk_sizes = dict(chunk_sizes or {})
    _chunk_sizes.setdefault("y", sizes["y"])
    _chunk_sizes.setdefault("x", sizes["x"])

    ordered_labels = [z for z in sizes if z not in "yx"]
    ordered_labels += ["y", "x"]
    dims = [
        Dimension(
            label=lbl,
            size=sizes[lbl],
            unit=DEFAULT_UNITS.get(lbl, None),
            chunk_size=_chunk_sizes.get(lbl, 1),
        )
        for lbl in cast("list[DimensionLabel]", ordered_labels)
    ]

    shape = [d.size for d in dims]
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):  # pragma: no cover
        raise ValueError(f"Unsupported dtype: {dtype}.  Must be an integer type.")

    # rng = np.random.default_rng()
    # data = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    data = np.ones(shape, dtype=dtype)

    def _build_plane_generator() -> Iterator[np.ndarray]:
        """Yield 2-D planes in y-x order."""
        i = 0
        if not (non_spatial_sizes := shape[:-2]):  # it's just a 2-D image
            yield data
        else:
            for idx in product(*(range(n) for n in non_spatial_sizes)):
                yield data[idx] * i
                i += 1

    return _build_plane_generator(), dims, dtype


def dims_from_useq(
    seq: useq.MDASequence,
    image_width: int,
    image_height: int,
    units: Mapping[str, UnitTuple | None] | None = None,
) -> list[Dimension]:
    """Convert a useq.MDASequence to a list of Dimensions for ome-writers.

    Parameters
    ----------
    seq : useq.MDASequence
        The `useq.MDASequence` to convert.
    image_width : int
        The expected width of the images in the stream.
    image_height : int
        The expected height of the images in the stream.
    units : Mapping[str, UnitTuple | None] | None, optional
        An optional mapping of dimension labels to their units. If `None`, defaults to
        - "t" -> (1.0, "s")
        - "z" -> (1.0, "um")
        - "y" -> (1.0, "um")
        - "x" -> (1.0, "um")

    Examples
    --------
    A typical usage of ome-writers with useq-schema might look like this:

    ```python
    from ome_writers import create_stream, dims_from_useq

    width, height = however_you_get_expected_image_dimensions()
    dims = dims_from_useq(seq, image_width=width, image_height=height)

    with create_stream(
        path=...,
        dimensions=dims,
        dtype=np.uint16,
        backend=...,
    ) as stream:
        for frame in whatever_generates_your_data():
            stream.append(frame)
    ```
    """
    try:
        from useq import MDASequence
    except ImportError:
        # if we can't import MDASequence, then seq must not be a MDASequence
        raise ValueError("seq must be a useq.MDASequence") from None
    else:
        if not isinstance(seq, MDASequence):
            raise ValueError("seq must be a useq.MDASequence")

    _units: Mapping[str, UnitTuple | None] = {
        **DEFAULT_UNITS,  # type: ignore[dict-item]
        **(units or {}),
    }

    dims: list[Dimension] = []
    for ax, size in seq.sizes.items():
        if size:
            # all of the useq axes are the same as the ones used here.
            dim_label = cast("DimensionLabel", str(ax))
            if dim_label not in _units:
                raise ValueError(f"Unsupported axis for OME: {ax}")
            dims.append(Dimension(label=dim_label, size=size, unit=_units[dim_label]))

    return [
        *dims,
        Dimension(label="y", size=image_height, unit=_units["y"]),
        Dimension(label="x", size=image_width, unit=_units["x"]),
    ]


def ngff_plate_and_wells_from_useq(
    seq: useq.MDASequence,
) -> tuple[PlateNGFF | None, dict[str, WellNGFF]]:
    """Convert a useq.MDASequence to both PlateNGFF and WellNGFF objects.

    Parameters
    ----------
    seq : useq.MDASequence
        The `useq.MDASequence` to convert.

    Returns
    -------
    tuple[PlateNGFF | None, dict[str, WellNGFF]]
        A tuple containing:
        - PlateNGFF object if the sequence contains plate information, otherwise None
        - Dictionary mapping well paths to WellNGFF objects (empty if no plate info)
    """
    try:
        from useq import MDASequence, WellPlatePlan
    except ImportError:
        return None, {}
    else:
        if not isinstance(seq, MDASequence):
            return None, {}

    # Check if stage_positions contains a WellPlatePlan
    stage_positions = seq.stage_positions
    if not isinstance(stage_positions, WellPlatePlan):
        return None, {}

    # Get plate information - create plate directly
    well_plate = stage_positions.plate

    # Create Row objects from plate rows
    row_names = [chr(ord("A") + i) for i in range(well_plate.rows)]
    col_names = [str(i + 1) for i in range(well_plate.columns)]

    rows = [RowNGFF(name=row_name) for row_name in row_names]
    columns = [ColumnNGFF(name=col_name) for col_name in col_names]

    # Create WellInPlate objects for selected wells only
    plate_wells = []
    selected_indices = stage_positions.selected_well_indices
    selected_names = stage_positions.selected_well_names

    for i, (row_idx, col_idx) in enumerate(selected_indices):
        well_name = selected_names[i]
        # Split the well name like "A1" into row "A" and column "1"
        row_name = well_name[0]  # First character is row
        col_name = well_name[1:]  # Rest is column
        well_path = f"{row_name}/{col_name}"

        plate_wells.append(
            WellInPlateNGFF(
                path=well_path, rowIndex=int(row_idx), columnIndex=int(col_idx)
            )
        )

    plate = PlateNGFF(
        columns=columns,
        rows=rows,
        wells=plate_wells,
        name=getattr(well_plate, "name", None),
        version="0.5",
    )

    # Create wells dictionary
    wells_dict = {}
    selected_indices = stage_positions.selected_well_indices
    selected_names = stage_positions.selected_well_names

    for i, (_row_idx, _col_idx) in enumerate(selected_indices):
        well_name = selected_names[i]
        # Convert well name "A1" to well path "A/1"
        row_name = well_name[0]  # First character is row
        col_name = well_name[1:]  # Rest is column
        well_path = f"{row_name}/{col_name}"

        # Create WellImageNGFF for this position
        images = [WellImageNGFF(path=str(i))]
        wells_dict[well_path] = WellNGFF(images=images, version="0.5")

    return plate, wells_dict
