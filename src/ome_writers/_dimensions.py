from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence


OME_DIM_TYPE = {"y": "space", "x": "space", "z": "space", "t": "time", "c": "channel"}
OME_UNIT = {"um": "micrometer", "ml": "milliliter", "s": "second", None: "unknown"}


# Recognized dimension labels
DimensionLabel: TypeAlias = Literal["x", "y", "z", "t", "c", "p", "other"]
# UnitTuple is a tuple of (scale, unit); e.g. (1, "s")
UnitTuple: TypeAlias = tuple[float, str]


class Dimension(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: DimensionLabel
    size: int
    unit: UnitTuple | None = None
    # None or 0 indicates no constraint.
    # -1 indicates that the chunk size should equal the full extent of the domain.
    chunk_size: int | None = None

    @property
    def ome_dim_type(self) -> Literal["space", "time", "channel", "other"]:
        return OME_DIM_TYPE.get(self.label, "other")  # type: ignore

    @property
    def ome_unit(self) -> str:
        if isinstance(self.unit, tuple):
            return OME_UNIT.get(self.unit[1], "unknown")
        return "unknown"

    @property
    def ome_scale(self) -> float:
        if isinstance(self.unit, tuple):
            return self.unit[0]
        return 1.0


def dims_to_ome_info(dims: Sequence[Dimension]) -> dict:
    """Convert a sequence of Dimension objects to OME dimensional information.

    This extracts dimensional information needed to build OME metadata.

    Parameters
    ----------
    dims : Sequence[Dimension]
        A sequence of Dimension objects describing the dimensions of the array.

    Returns
    -------
    dict
        Dictionary containing:
        - 'sizes': dict mapping dimension labels to sizes
        - 'n_positions': number of positions
    """
    # Find the position dimension, if any
    if any(dim.label not in "tczyxp" for dim in dims):
        raise NotImplementedError("Only dimensions t, c, z, y, x, and p are supported.")

    dims_sizes = {dim.label: dim.size for dim in dims}
    n_positions = dims_sizes.pop("p", 1)

    return {
        "sizes": dims_sizes,
        "n_positions": n_positions,
    }


def dims_to_ngff_axes(dims: Sequence[Dimension]) -> tuple[list[dict], list[float]]:
    """Convert a sequence of Dimension objects to NGFF axes and scales.

    The length of "axes" must be between 2 and 5 and MUST be equal to the
    dimensionality of the zarr arrays storing the image data. The "axes" MUST
    contain 2 or 3 entries of "type:space" and MAY contain one additional
    entry of "type:time" and MAY contain one additional entry of
    "type:channel" or a null / custom type. The order of the entries MUST
    correspond to the order of dimensions of the zarr arrays. In addition, the
    entries MUST be ordered by "type" where the "time" axis must come first
    (if present), followed by the "channel" or custom axis (if present) and
    the axes of type "space".

    Parameters
    ----------
    dims : Sequence[Dimension]
        A sequence of Dimension objects describing the dimensions of the array.

    Returns
    -------
    tuple[list[dict], list[float]]
        A tuple containing:
        - axes: List of axis dictionaries with "name", "type", and "unit" keys
        - scales: List of scale values for each dimension
    """
    axes: list[dict] = []
    scales: list[float] = []
    for dim in dims:
        axes.append(
            {"name": dim.label, "type": dim.ome_dim_type, "unit": dim.ome_unit},
        )
        scales.append(dim.ome_scale)
    return axes, scales
