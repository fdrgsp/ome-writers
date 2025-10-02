from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._ngff_metadata import ngff_meta_v5
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from yaozarrs.v05 import Plate

    from ome_writers._dimensions import Dimension


class TensorStoreZarrStream(MultiPositionOMEStream):
    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover
        """Check if the tensorstore package is available."""
        return importlib.util.find_spec("tensorstore") is not None

    def __init__(self) -> None:
        try:
            import tensorstore
        except ImportError as e:
            msg = (
                "TensorStoreZarrStream requires tensorstore: `pip install tensorstore`."
            )
            raise ImportError(msg) from e

        self._ts = tensorstore
        super().__init__()
        self._group_path: Path
        self._array_paths: dict[str, Path] = {}  # array_key -> path mapping
        self._futures: list = []
        self._stores: dict[str, tensorstore.TensorStore] = {}  # array_key -> store
        self._delete_existing = True

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        plate: Plate | None = None,
        *,
        overwrite: bool = False,
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic with HCS support
        _, non_position_dims = self._init_positions(dimensions, plate=plate)
        self._delete_existing = overwrite

        self._create_group(self._normalize_path(path), dimensions)

        # Create stores for each unique array key
        unique_array_keys = set()
        for pos_idx in range(len(self._indices)):
            array_key, _ = self._indices[pos_idx]
            unique_array_keys.add(array_key)

        for array_key in unique_array_keys:
            spec = self._create_spec(dtype, non_position_dims, array_key)
            try:
                self._stores[array_key] = self._ts.open(spec).result()
            except ValueError as e:
                if "ALREADY_EXISTS" in str(e):
                    raise FileExistsError(
                        f"Array {array_key} already exists at "
                        f"{self._array_paths[array_key]}. "
                        "Use overwrite=True to overwrite it."
                    ) from e
                else:
                    raise

        # Now create the group metadata after all arrays are set up
        self._create_group_metadata()
        return self

    def _create_spec(
        self, dtype: np.dtype, dimensions: Sequence[Dimension], array_key: str
    ) -> dict:
        labels = tuple(d.label for d in dimensions)
        shape = tuple(d.size for d in dimensions)
        units = tuple(d.unit for d in dimensions)
        chunk_shape = tuple(d.chunk_size for d in dimensions)
        return {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self._array_paths[array_key])},
            "schema": {
                "domain": {"shape": shape, "labels": labels},
                "dtype": dtype.name,
                "chunk_layout": {"chunk": {"shape": chunk_shape}},
                "dimension_units": units,
            },
            "create": True,
            "delete_existing": self._delete_existing,
        }

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TensorStore-specific write implementation."""
        store = self._stores[array_key]
        future = store[index].write(frame)  # type: ignore[index]
        self._futures.append(future)

    def flush(self) -> None:
        # Wait for all writes to finish.
        for future in self._futures:
            future.result()
        self._futures.clear()
        self._stores.clear()

    def is_active(self) -> bool:
        return bool(self._stores)

    def _create_group(self, path: str, dims: Sequence[Dimension]) -> Path:
        self._group_path = Path(path)
        # Remove existing directory if delete_existing is True
        if self._delete_existing and self._group_path.exists():
            import shutil

            shutil.rmtree(self._group_path)

        self._group_path.mkdir(parents=True, exist_ok=True)

        # Set up array paths for all unique array keys - TensorStore will create them
        unique_array_keys = set()
        for pos_idx in range(len(self._indices)):
            array_key, _ = self._indices[pos_idx]
            unique_array_keys.add(array_key)

        for array_key in unique_array_keys:
            self._array_paths[array_key] = self._group_path / array_key

        # We'll create the group metadata after all arrays are created
        return self._group_path

    def _create_group_metadata(self) -> None:
        """Create the group-level zarr.json with NGFF metadata after arrays exist."""
        # Use the array keys and dimensions from the parent's HCS-aware logic
        array_dims: dict[str, Sequence[Dimension]] = {}
        unique_array_keys = set()
        for pos_idx in range(len(self._indices)):
            array_key, _ = self._indices[pos_idx]
            unique_array_keys.add(array_key)

        for array_key in unique_array_keys:
            # Use non-position dimensions for the array metadata
            array_dims[array_key] = self._non_position_dims

        group_zarr = self._group_path / "zarr.json"
        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": ngff_meta_v5(
                array_dims=array_dims,
                plate=self._plate,
            ),
        }
        group_zarr.write_text(json.dumps(group_meta, indent=2))

        # Create well-level metadata files if this is HCS data
        if self._is_hcs_data():
            self._create_well_metadata_files()

    def _create_well_metadata_files(self) -> None:
        """Create individual zarr.json metadata files for each well directory."""
        # Group arrays by well path
        well_arrays: dict[str, list[str]] = {}
        for array_key in self._get_unique_array_keys():
            # Extract well path from array key (e.g., "A/1/0" -> "A/1")
            parts = array_key.split("/")
            if len(parts) >= 2:
                well_path = "/".join(parts[:2])  # "A/1"
                well_arrays.setdefault(well_path, []).append(array_key)

        # Create metadata file for each well
        for well_path, array_keys in well_arrays.items():
            well_metadata = self._wells.get(well_path)
            if well_metadata is None:
                continue

            # Create array metadata for this well's arrays
            well_array_meta = dict.fromkeys(array_keys, self._non_position_dims)

            # Generate well-specific metadata
            well_attrs = ngff_meta_v5(
                well_array_meta,
                plate=None,  # No plate metadata at well level
            )

            # Write well metadata file
            well_dir = Path(self._group_path) / well_path
            well_dir.mkdir(parents=True, exist_ok=True)
            well_zarr_json = well_dir / "zarr.json"

            well_meta = {
                "consolidated_metadata": None,
                "node_type": "group",
                "zarr_format": 3,
                "attributes": well_attrs,
            }
            well_zarr_json.write_text(json.dumps(well_meta, indent=2))

    def _get_unique_array_keys(self) -> set[str]:
        """Get all unique array keys from the indices mapping."""
        unique_keys = set()
        for array_key, _ in self._indices.values():
            unique_keys.add(array_key)
        return unique_keys
