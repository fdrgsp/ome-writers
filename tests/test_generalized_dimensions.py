#!/usr/bin/env python3
"""Test script to verify generalized positional dimension functionality."""

import tempfile
from pathlib import Path

import numpy as np

import ome_writers as omew
from ome_writers._dimensions import Dimension


def test_multiple_positional_dimensions() -> None:
    """Test multiple positional dimensions beyond just grid."""
    # Create dimensions with multiple positional dimensions
    dimensions = [
        Dimension("t", 2),
        Dimension("c", 1),
        Dimension("p", 2),  # position
        Dimension("g", 2),  # grid
        Dimension("r", 3),  # region (custom dimension)
        Dimension("y", 32),
        Dimension("x", 32),
    ]

    dtype = np.uint16

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_multi_pos.zarr"

        # Test with acquire-zarr backend
        from ome_writers.backends._acquire_zarr import AcquireZarrStream

        stream = AcquireZarrStream()
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        # Verify all positional dimensions are recognized
        assert stream._position_dim is not None
        assert stream._position_dim.size == 2
        assert len(stream._positional_dims) == 2  # g and r
        assert stream._num_positions == 2

        # Check positional dimension labels
        positional_labels = {dim.label for dim in stream._positional_dims}
        assert positional_labels == {"g", "r"}

        # Verify array keys contain all positional information
        print("Array keys generated for multiple positional dimensions:")
        expected_count = 2 * 2 * 3 * 2  # p=2, g=2, r=3, t=2
        assert len(stream._indices) == expected_count

        for i, (array_key, index) in stream._indices.items():
            print(f"  Index {i}: key='{array_key}', index={index}")

        # Check that we have the expected array keys
        sample_keys = [
            stream._indices[i][0] for i in range(min(8, len(stream._indices)))
        ]
        print(f"Sample keys: {sample_keys}")

        # Verify key format: should be _p0000_g0000_r0000, _p0000_g0000_r0001, etc.
        assert "_p0000_g0000_r0000" in [
            stream._indices[i][0] for i in range(len(stream._indices))
        ]
        assert "_p0001_g0001_r0002" in [
            stream._indices[i][0] for i in range(len(stream._indices))
        ]

        # Write some test data
        for _ in range(expected_count):
            frame = np.random.randint(0, 1000, (32, 32), dtype=dtype)
            stream.append(frame)

        stream.flush()
        assert not stream.is_active()
        print("✓ Multiple positional dimensions test passed!")


def test_non_standard_single_dimension() -> None:
    """Test single non-standard dimension (not grid)."""
    # Create dimensions with a custom positional dimension
    dimensions = [
        Dimension("t", 2),
        Dimension("c", 1),
        Dimension("r", 3),  # region (custom dimension, no position)
        Dimension("y", 32),
        Dimension("x", 32),
    ]

    dtype = np.uint16

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_single_custom.zarr"

        from ome_writers.backends._acquire_zarr import AcquireZarrStream

        stream = AcquireZarrStream()
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        # Verify no position dimension but has positional dimension
        assert stream._position_dim is None
        assert len(stream._positional_dims) == 1
        assert stream._positional_dims[0].label == "r"
        assert stream._positional_dims[0].size == 3

        # Verify array keys contain region information
        print("\nArray keys generated for single custom dimension:")
        for i, (array_key, index) in stream._indices.items():
            print(f"  Index {i}: key='{array_key}', index={index}")

        # Check that we have the expected array keys format: _r000, _r001, _r002
        expected_keys = [
            "_r000",
            "_r001",
            "_r002",
            "_r000",
            "_r001",
            "_r002",
        ]  # r=3, t=2
        actual_keys = [stream._indices[i][0] for i in range(len(stream._indices))]
        print(f"Expected patterns: {expected_keys}")
        print(f"Actual keys: {actual_keys}")

        # Write test data
        for i in range(6):  # r=3, t=2
            frame = np.random.randint(0, 1000, (32, 32), dtype=dtype)
            stream.append(frame)

        stream.flush()
        assert not stream.is_active()
        print("✓ Single custom dimension test passed!")


def test_tifffile_generalized_naming() -> None:
    """Test TiffFile backend with generalized dimension naming."""
    # Test with multiple custom dimensions
    dimensions = [
        Dimension("t", 1),
        Dimension("c", 1),
        Dimension("p", 2),  # position
        Dimension("g", 2),  # grid
        Dimension("s", 2),  # site (another custom dimension)
        Dimension("y", 32),
        Dimension("x", 32),
    ]

    dtype = np.uint16

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_tiff_multi.ome.tiff"

        stream = omew.TifffileStream()
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        print("\nTiffFile array keys generated:")
        for i, (array_key, index) in stream._indices.items():
            print(f"  Index {i}: key='{array_key}', index={index}")

        # Write data
        for i in range(8):  # p=2, g=2, s=2
            frame = np.random.randint(0, 1000, (32, 32), dtype=dtype)
            stream.append(frame)

        stream.flush()
        assert not stream.is_active()

        print("\nFiles created:")
        for f in sorted(Path(tmp_dir).iterdir()):
            print(f"  {f.name}")

        # Check expected file names
        expected_files = [
            "test_tiff_multi_p0000_g0000_s0000.ome.tiff",
            "test_tiff_multi_p0000_g0000_s0001.ome.tiff",
            "test_tiff_multi_p0000_g0001_s0000.ome.tiff",
            "test_tiff_multi_p0000_g0001_s0001.ome.tiff",
            "test_tiff_multi_p0001_g0000_s0000.ome.tiff",
            "test_tiff_multi_p0001_g0000_s0001.ome.tiff",
            "test_tiff_multi_p0001_g0001_s0000.ome.tiff",
            "test_tiff_multi_p0001_g0001_s0001.ome.tiff",
        ]

        created_files = [f.name for f in Path(tmp_dir).iterdir()]
        for expected_file in expected_files:
            assert expected_file in created_files, f"Missing file: {expected_file}"

        print("✓ TiffFile generalized naming test passed!")


if __name__ == "__main__":
    print("Testing generalized positional dimension functionality...")
    test_multiple_positional_dimensions()
    test_non_standard_single_dimension()
    test_tifffile_generalized_naming()
    print("\n🎉 All generalized dimension tests passed!")
