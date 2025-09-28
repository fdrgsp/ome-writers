"""OME metadata construction utilities."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ome_writers import __version__
from ome_writers.model import dims_to_ome_info

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ome_types

    from ome_writers.model import Dimension


def ome_meta(
    dims: Sequence[Dimension],
    dtype: np.typing.DTypeLike,
    tiff_file_name: str | None = None,
) -> ome_types.OME:
    """Create an OME object from dimension objects.

    Parameters
    ----------
    dims : Sequence[Dimension]
        A sequence of Dimension objects describing the dimensions of the array.
    dtype : np.typing.DTypeLike
        Data type for the image data
    tiff_file_name : str | None
        Optional TIFF file name for metadata

    Returns
    -------
    ome_types.OME
        Complete OME metadata object
    """
    try:
        from ome_types import model as m
    except ImportError as e:
        raise ImportError(
            "The `ome-types` package is required to use this function. "
            "Please install it via `pip install ome-types` or use the `tiff` extra."
        ) from e

    dimensions_info = dims_to_ome_info(dims)
    dims_sizes = dimensions_info["sizes"]
    n_positions = dimensions_info["n_positions"]

    # Determine dimension order
    _dim_names = "".join(reversed(dims_sizes)).upper()
    dim_order = next(
        (x for x in m.Pixels_DimensionOrder if x.value.startswith(_dim_names)),
        m.Pixels_DimensionOrder.XYCZT,
    )

    # Create channels
    channels = [
        m.Channel(
            id=f"Channel:{i}",
            name=f"Channel {i + 1}",
            samples_per_pixel=1,  # TODO
        )
        for i in range(dims_sizes.get("c", 0))
    ]

    uuid_ = f"urn:uuid:{uuid.uuid4()}"
    images: list[m.Image] = []

    for p in range(n_positions):
        planes: list[m.Plane] = []
        tiff_blocks: list[m.TiffData] = []
        ifd = 0

        # Create planes for all TCZ combinations
        tcz_sizes = {k: v for k, v in dims_sizes.items() if k in "tcz"}
        if tcz_sizes:
            labels, sizes = zip(*tcz_sizes.items(), strict=False)
            has_z, has_t, has_c = "z" in labels, "t" in labels, "c" in labels

            for index in np.ndindex(*sizes):
                plane = m.Plane(
                    the_z=index[labels.index("z")] if has_z else 0,
                    the_t=index[labels.index("t")] if has_t else 0,
                    the_c=index[labels.index("c")] if has_c else 0,
                )
                planes.append(plane)

                if tiff_file_name is not None:
                    tiff_data = m.TiffData(
                        ifd=ifd,
                        uuid=m.TiffData.UUID(value=uuid_, file_name=tiff_file_name),
                        first_c=plane.the_c,
                        first_z=plane.the_z,
                        first_t=plane.the_t,
                        plane_count=1,
                    )
                    tiff_blocks.append(tiff_data)
                ifd += 1

        md_only = None if tiff_blocks else m.MetadataOnly()
        pix_type = m.PixelType(np.dtype(dtype).name)

        pixels = m.Pixels(
            id=f"Pixels:{p}",
            channels=channels,
            planes=planes,
            tiff_data_blocks=tiff_blocks,
            metadata_only=md_only,
            dimension_order=dim_order,
            type=pix_type,
            # significant_bits=..., # TODO
            size_x=dims_sizes.get("x", 1),
            size_y=dims_sizes.get("y", 1),
            size_z=dims_sizes.get("z", 1),
            size_c=dims_sizes.get("c", 1),
            size_t=dims_sizes.get("t", 1),
            # physical_size_x=voxel_size.x,
            # physical_size_y=voxel_size.y,
            # physical_size_z = voxel_size.z
            # physical_size_x_unit=UnitsLength.MICROMETER,
            # physical_size_y_unit=UnitsLength.MICROMETER,
            # physical_size_z_unit = UnitsLength.MICROMETER
        )

        base_name = Path(tiff_file_name).stem if tiff_file_name else f"Image_{p}"
        images.append(
            m.Image(
                # objective_settings=...
                id=f"Image:{p}",
                name=base_name + (f" (Series {p})" if n_positions > 1 else ""),
                pixels=pixels,
                # acquisition_date=acquisition_date,
            )
        )

    ome = m.OME(images=images, creator=f"ome_writers v{__version__}")
    return ome
