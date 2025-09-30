"""Tests for ome-writers library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    from .conftest import AvailableBackend


def test_minimal_2d_dimensions(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test with minimal 2D dimensions (just x and y)."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    # Create the stream
    stream = backend.cls()

    # Set output path
    output_path = tmp_path / f"test_2d_{backend.name.lower()}.{backend.file_ext}"

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Get the data from the generator
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert not stream.is_active()
    assert output_path.exists()


def test_stream_error_handling(backend: AvailableBackend) -> None:
    """Test error handling in streams."""
    empty_stream = backend.cls()

    expected_message = "Stream is closed or uninitialized"
    test_frame = np.zeros((64, 64), dtype=np.uint16)
    with pytest.raises(RuntimeError, match=expected_message):
        empty_stream.append(test_frame)


def test_dimension_info_properties() -> None:
    """Test DimensionInfo properties."""
    # Test spatial dimension
    x_dim = omew.Dimension(label="x", size=100, unit=(0.5, "um"), chunk_size=50)
    assert x_dim.ome_dim_type == "space"
    assert x_dim.ome_unit == "micrometer"
    assert x_dim.ome_scale == 0.5

    # Test time dimension
    t_dim = omew.Dimension(label="t", size=10, unit=(2.0, "s"), chunk_size=1)
    assert t_dim.ome_dim_type == "time"
    assert t_dim.ome_unit == "second"
    assert t_dim.ome_scale == 2.0

    # Test channel dimension
    c_dim = omew.Dimension(label="c", size=3, chunk_size=1)
    assert c_dim.ome_dim_type == "channel"
    assert c_dim.ome_unit == "unknown"
    assert c_dim.ome_scale == 1.0

    # Test custom dimension
    p_dim = omew.Dimension(label="p", size=5, chunk_size=1)
    assert p_dim.ome_dim_type == "other"


def test_create_stream_factory_function(
    backend: AvailableBackend, tmp_path: Path
) -> None:
    """Test the create_stream factory function."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "z": 2, "c": 2, "y": 64, "x": 64},
        chunk_sizes={"y": 32, "x": 32},
    )

    output_path = tmp_path / f"factory_test.{backend.file_ext}"
    stream = omew.create_stream(
        str(output_path), dtype, dimensions, backend=backend.name
    )
    assert isinstance(stream, omew.OMEStream)
    assert stream.is_active()

    # Write a test frame
    for data in data_gen:
        stream.append(data)
        break  # Just write one frame for this test

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()


@pytest.mark.parametrize(
    "dtype", [np.dtype(np.uint8), np.dtype(np.uint16)], ids=["uint8", "uint16"]
)
def test_data_integrity_roundtrip(
    backend: AvailableBackend,
    tmp_path: Path,
    dtype: np.dtype,
) -> None:
    """Test data integrity roundtrip with different data types."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "z": 2, "c": 2, "y": 64, "x": 64},
        chunk_sizes={"y": 32, "x": 32},
    )

    # Convert generator to list to use data multiple times
    original_frames = list(data_gen)
    output_path = tmp_path / f"{backend.name.lower()}_{dtype.name}{backend.file_ext}"

    # Write data using our stream
    stream = backend.cls()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()
    for frame in original_frames:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    # Read data back and verify it matches
    assert output_path.exists()
    disk_data = backend.read_data(output_path)

    # Reconstruct original data array from frames
    shape = tuple(d.size for d in dimensions)
    original_data = np.array(original_frames).reshape(shape)

    # Verify the data matches exactly
    np.testing.assert_array_equal(
        original_data,
        disk_data,
        err_msg=f"Data mismatch in {backend.name} roundtrip test with {dtype}",
    )

    # Test 2: Try to create again without overwrite (should fail)
    with pytest.raises(FileExistsError, match=r".*already exists"):
        stream = backend.cls()
        stream = stream.create(str(output_path), dtype, dimensions, overwrite=False)

    # Test 3: Create again with overwrite=True (should succeed)
    stream = backend.cls()
    stream = stream.create(str(output_path), dtype, dimensions, overwrite=True)
    assert isinstance(stream, omew.OMEStream)
    assert stream.is_active()

    for frame in original_frames:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()


def test_multiposition_acquisition(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test multi-position acquisition support with position dimension."""
    stream_cls = backend.cls
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "z": 2, "c": 2, "y": 32, "x": 32, "p": 3},
        chunk_sizes={"y": 16, "x": 16},
    )

    # Create the stream
    stream = stream_cls()
    output_path = (
        tmp_path / f"test_multipos_{stream_cls.__name__.lower()}.{backend.file_ext}"
    )

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write frames for all positions and time/channel combinations
    # Total frames = 3 positions x 2 time x 2 channels = 12 frames
    for frame in data_gen:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    if backend.file_ext.endswith("zarr"):
        assert output_path.exists()
        # Verify zarr structure
        assert (output_path / "0").exists()
        assert (output_path / "1").exists()
        assert (output_path / "2").exists()
        assert (output_path / "zarr.json").exists()

        # Verify that each position has correct metadata
        with open(output_path / "zarr.json") as f:
            group_meta = json.load(f)

        ome_attrs = group_meta["attributes"]["ome"]
        multiscales = ome_attrs["multiscales"]
        assert ome_attrs["version"] == "0.5"
        assert isinstance(multiscales, list)
        assert len(multiscales) == 1
        assert len(multiscales[0]["datasets"]) == 3

        axes_names = {ax["name"] for ax in multiscales[0]["axes"]}
        assert all(x in axes_names for x in ["t", "c", "y", "x"])

    elif (ext := backend.file_ext).endswith("tiff"):
        # For TIFF, separate files are created for each position
        base_path = Path(str(output_path).replace(ext, ""))
        assert (base_path.with_name(f"{base_path.name}_p000{ext}")).exists()
        assert (base_path.with_name(f"{base_path.name}_p001{ext}")).exists()
        assert (base_path.with_name(f"{base_path.name}_p002{ext}")).exists()

        # Verify that each TIFF file has the correct metadata and shape
        for pos_idx in range(3):
            pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}{ext}")
            assert pos_file.exists()

            # Read the file to verify it has correct shape
            data = backend.read_data(pos_file)
            # Shape should be (t, z, c, y, x) = (3, 2, 2, 32, 32)
            expected_shape = (3, 2, 2, 32, 32)
            assert data.shape == expected_shape


def test_hcs_plate_only_metadata(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test HCS acquisition using only plate metadata (no wells parameter)."""
    from ome_writers.model._hcs import (
        ColumnNGFF,
        PlateNGFF,
        RowNGFF,
        WellInPlateNGFF,
    )

    # Create test plate with 2 wells
    rows = [RowNGFF(name="A")]
    columns = [ColumnNGFF(name="1"), ColumnNGFF(name="2")]
    wells = [
        WellInPlateNGFF(path="A/1", rowIndex=0, columnIndex=0),
        WellInPlateNGFF(path="A/2", rowIndex=0, columnIndex=1),
    ]
    plate = PlateNGFF(columns=columns, rows=rows, wells=wells, version="0.5")

    # Create dimensions with position dimension (4 positions = 2 per well)
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"p": 4, "c": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / f"test_hcs_{backend.name.lower()}.{backend.file_ext}"

    # Test that create_stream works with only plate parameter (no wells)
    stream = omew.create_stream(
        str(output_path),
        dtype,
        dimensions,
        plate=plate,  # Only plate metadata, no wells parameter
        backend=backend.name,
        overwrite=True,
    )

    assert stream.is_active()

    # Write all frames
    for frame in data_gen:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    # Verify that HCS structure is created correctly
    if backend.file_ext.endswith("zarr"):
        assert output_path.exists()
        # Check that well directories exist
        assert (output_path / "A" / "1").exists()
        assert (output_path / "A" / "2").exists()

        # Verify plate metadata exists
        with open(output_path / "zarr.json") as f:
            group_meta = json.load(f)

        ome_attrs = group_meta["attributes"]["ome"]
        assert "plate" in ome_attrs
        plate_meta = ome_attrs["plate"]
        assert len(plate_meta["wells"]) == 2
        assert plate_meta["wells"][0]["path"] == "A/1"
        assert plate_meta["wells"][1]["path"] == "A/2"

    elif backend.file_ext.endswith("tiff"):
        # For TIFF with HCS data, separate files are created for each position
        base_path = Path(str(output_path).replace(backend.file_ext, ""))
        ext = backend.file_ext
        assert (base_path.with_name(f"{base_path.name}_p000{ext}")).exists()
        assert (base_path.with_name(f"{base_path.name}_p001{ext}")).exists()
        assert (base_path.with_name(f"{base_path.name}_p002{ext}")).exists()
        assert (base_path.with_name(f"{base_path.name}_p003{ext}")).exists()


def test_multi_fov_per_well_distribution() -> None:
    """Test that multiple FOVs are properly distributed across wells."""
    from ome_writers._stream_base import MultiPositionOMEStream
    from ome_writers.model._hcs import (
        ColumnNGFF,
        PlateNGFF,
        RowNGFF,
        WellInPlateNGFF,
    )

    # Create a concrete implementation for testing
    class TestStream(MultiPositionOMEStream):
        def create(
            self,
            path: str,
            dtype: np.dtype,
            dimensions: list,
            plate: PlateNGFF | None = None,
            *,
            overwrite: bool = False,
        ) -> MultiPositionOMEStream:
            return self

        def _write_to_backend(
            self, array_key: str, index: tuple[int, ...], frame: np.ndarray
        ) -> None:
            pass

        def is_active(self) -> bool:
            return True

        def flush(self) -> None:
            pass

        @classmethod
        def is_available(cls) -> bool:
            return True

    # Test case 1: 7 positions across 3 wells should distribute as [3, 2, 2]
    rows = [RowNGFF(name="A")]
    columns = [ColumnNGFF(name="1"), ColumnNGFF(name="2"), ColumnNGFF(name="3")]
    wells = [
        WellInPlateNGFF(path="A/1", rowIndex=0, columnIndex=0),
        WellInPlateNGFF(path="A/2", rowIndex=0, columnIndex=1),
        WellInPlateNGFF(path="A/3", rowIndex=0, columnIndex=2),
    ]
    plate = PlateNGFF(columns=columns, rows=rows, wells=wells, version="0.5")

    stream = TestStream()
    wells_dict = stream._extract_wells_from_plate(plate, 7)

    assert len(wells_dict) == 3
    assert len(wells_dict["A/1"].images) == 3  # Gets extra field
    assert len(wells_dict["A/2"].images) == 2
    assert len(wells_dict["A/3"].images) == 2

    # Verify field indices are sequential
    all_field_indices = []
    for well_path in ["A/1", "A/2", "A/3"]:
        for image in wells_dict[well_path].images:
            all_field_indices.append(int(image.path))

    assert sorted(all_field_indices) == list(range(7))  # [0, 1, 2, 3, 4, 5, 6]

    # Test case 2: 6 positions across 2 wells should distribute as [3, 3]
    wells_2 = [
        WellInPlateNGFF(path="A/1", rowIndex=0, columnIndex=0),
        WellInPlateNGFF(path="A/2", rowIndex=0, columnIndex=1),
    ]
    plate_2 = PlateNGFF(columns=columns[:2], rows=rows, wells=wells_2, version="0.5")

    wells_dict_2 = stream._extract_wells_from_plate(plate_2, 6)

    assert len(wells_dict_2) == 2
    assert len(wells_dict_2["A/1"].images) == 3
    assert len(wells_dict_2["A/2"].images) == 3


def test_ngff_meta_v5_wells_parameter_removed() -> None:
    """Test that ngff_meta_v5 function no longer accepts wells parameter."""
    import inspect

    from ome_writers._ngff_metadata import ngff_meta_v5

    # Check function signature
    sig = inspect.signature(ngff_meta_v5)
    param_names = list(sig.parameters.keys())

    # Verify wells and well_path parameters are not in the signature
    assert "wells" not in param_names
    assert "well_path" not in param_names
    assert "plate" in param_names

    # Verify the function works with just array_dims and plate
    from ome_writers.model import Dimension

    dimensions = [
        Dimension(label="t", size=1),
        Dimension(label="c", size=1),
        Dimension(label="y", size=64),
        Dimension(label="x", size=64),
    ]

    array_dims = {"0": dimensions}

    # Should work without any HCS metadata
    metadata = ngff_meta_v5(array_dims)
    assert "ome" in metadata
    assert "multiscales" in metadata["ome"]

    # Should work with plate metadata
    from ome_writers.model._hcs import (
        ColumnNGFF,
        PlateNGFF,
        RowNGFF,
        WellInPlateNGFF,
    )

    rows = [RowNGFF(name="A")]
    columns = [ColumnNGFF(name="1")]
    wells = [WellInPlateNGFF(path="A/1", rowIndex=0, columnIndex=0)]
    plate = PlateNGFF(columns=columns, rows=rows, wells=wells, version="0.5")

    metadata_with_plate = ngff_meta_v5(array_dims, plate=plate)
    assert "plate" in metadata_with_plate["ome"]
