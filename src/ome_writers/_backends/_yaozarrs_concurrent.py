"""yaozarrs v0.5 `PlateBuilder` and `Bf2RawBuilder` that create groups concurrently.

yaozarrs creates the hierarchy one node at a time: for a plate with hundreds of
fields that is ~1000 sequential `mkdir`s and `zarr.json` writes, which is slow
wherever file operations have latency (hard disks, network drives, Windows).
The output is the same, byte for byte.

Stopgap until this lands in yaozarrs: it uses yaozarrs' private helpers, and falls
back to yaozarrs' own `prepare()` where they don't exist.
"""

# pyright: reportPossiblyUnboundVariable=false

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from yaozarrs.write.v05 import Bf2RawBuilder, PlateBuilder

from ome_writers._util import thread_map

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

try:
    from yaozarrs.write._common._arrays import _get_create_func
    from yaozarrs.write._common._builders import _prepare_image
    from yaozarrs.write._common._groups import _create_zarr3_group
    from yaozarrs.write._common._plate import _merge_plate_metadata

    _HAVE_PRIVATE_HELPERS = True
except ImportError:  # pragma: no cover
    _HAVE_PRIVATE_HELPERS = False


class ConcurrentBf2RawBuilder(Bf2RawBuilder):
    def prepare(self) -> tuple[Path, dict[str, Any]]:
        if not _HAVE_PRIVATE_HELPERS:  # pragma: no cover
            return super().prepare()

        bf2raw = self._bf2raw_type(bioformats2raw_layout=3)
        _create_zarr3_group(
            self._dest, bf2raw, self._overwrite, extra_attributes=self._extra_attributes
        )
        ome_path = self._dest / "OME"
        series = self._series_type(series=list(self._series))
        _create_zarr3_group(ome_path, series, self._overwrite)
        if self._ome_xml is not None:  # pragma: no cover
            (ome_path / "METADATA.ome.xml").write_text(self._ome_xml)
        return self._dest, _prepare_images(self, self._series)


class ConcurrentPlateBuilder(PlateBuilder):
    def prepare(self) -> tuple[Path, dict[str, Any]]:
        if not _HAVE_PRIVATE_HELPERS:  # pragma: no cover
            return super().prepare()

        plate = _merge_plate_metadata(
            self._get_images_dict(), self._user_plate, self._plate_type
        )
        _create_zarr3_group(
            self._dest, plate, self._overwrite, extra_attributes=self._extra_attributes
        )

        # zarr v3 has no implicit groups: create rows, then wells, then fields
        def create_row(row: str) -> None:
            (self._dest / row).mkdir()
            group = {"zarr_format": 3, "node_type": "group"}
            (self._dest / row / "zarr.json").write_text(json.dumps(group))

        def create_well(path: str) -> None:
            well = self._generate_well_metadata(list(self._wells[path]))
            _create_zarr3_group(self._dest / path, well)

        thread_map(create_row, {path.split("/")[0] for path in self._wells})
        thread_map(create_well, self._wells)
        fields = {
            f"{path}/{fov}": image
            for path, fovs in self._wells.items()
            for fov, image in fovs.items()
        }
        return self._dest, _prepare_images(self, fields)


def _prepare_images(
    builder: ConcurrentBf2RawBuilder | ConcurrentPlateBuilder,
    images: Mapping[str, tuple[Any, Any]],
) -> dict[str, Any]:
    """Create each `{path: (image, datasets)}` group and its arrays, concurrently.

    Returns `{f"{path}/{dataset_path}": array}`.
    """
    create_func = _get_create_func(builder._writer)  # once, not per image

    def prepare(path: str) -> dict[str, Any]:
        image, datasets = images[path]
        _, arrays = _prepare_image(
            builder._dest / path,
            image,
            datasets,
            image_type=builder._image_type,
            chunks=builder._chunks,
            shards=builder._shards,
            writer=create_func,
            # the root was just created, so there is nothing to delete
            overwrite=False,
            compression=builder._compression,
        )
        return {f"{path}/{key}": array for key, array in arrays.items()}

    return {k: v for arrays in thread_map(prepare, images) for k, v in arrays.items()}
