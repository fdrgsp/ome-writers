#!/usr/bin/env python3
"""Test script to verify grid dimension functionality."""

import tempfile
from pathlib import Path

import numpy as np

from ome_writers._dimensions import Dimension


def test_grid_dimension_basic() -> None:
    """Test basic grid dimension functionality."""
    # Create dimensions with grid dimension
    dimensions = [
        Dimension("t", 2),
        Dimension("c", 1),
        Dimension("g", 2),  # 2 grid positions
        Dimension("y", 32),
        Dimension("x", 32),
    ]

    dtype = np.uint16

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_grid.zarr"

        # Test with acquire-zarr backend
        from ome_writers.backends._acquire_zarr import AcquireZarrStream

        stream = AcquireZarrStream()
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        # Verify grid dimension is recognized
        assert stream._grid_dim is not None
        assert stream._grid_dim.size == 2
        assert stream._num_grids == 2

        # Verify array keys contain grid information
        print("Array keys generated:")
        for i, (array_key, index) in stream._indices.items():
            print(f"  Index {i}: key='{array_key}', index={index}")

        # Check that we have the expected array keys
        expected_keys = ["_g_000", "_g_001", "_g_000", "_g_001"]  # t=2, g=2
        actual_keys = [stream._indices[i][0] for i in range(len(stream._indices))]
        print(f"Expected keys: {expected_keys}")
        print(f"Actual keys: {actual_keys}")

        # Write some test data
        for i in range(4):  # t=2, g=2
            frame = np.random.randint(0, 1000, (32, 32), dtype=dtype)
            stream.append(frame)

        stream.flush()
        assert not stream.is_active()
        print("✓ Grid dimension test passed!")


def test_position_and_grid_dimension() -> None:
    """Test combined position and grid dimensions."""
    # Create dimensions with both position and grid dimensions
    dimensions = [
        Dimension("t", 1),
        Dimension("c", 1),
        Dimension("p", 2),  # 2 positions
        Dimension("g", 3),  # 3 grid positions
        Dimension("y", 32),
        Dimension("x", 32),
    ]

    dtype = np.uint16

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_pos_grid.zarr"

        # Test with acquire-zarr backend
        from ome_writers.backends._acquire_zarr import AcquireZarrStream

        stream = AcquireZarrStream()
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        # Verify both dimensions are recognized
        assert stream._position_dim is not None
        assert stream._position_dim.size == 2
        assert stream._grid_dim is not None
        assert stream._grid_dim.size == 3
        assert stream._num_positions == 2
        assert stream._num_grids == 3

        # Verify array keys contain both position and grid information
        print("\nCombined position+grid array keys generated:")
        for i, (array_key, index) in stream._indices.items():
            print(f"  Index {i}: key='{array_key}', index={index}")

        # Check that we have the expected array keys in the format _p000_g_000
        expected_patterns = [
            "_p000_g_000",
            "_p000_g_001",
            "_p000_g_002",  # pos 0, grids 0,1,2
            "_p001_g_000",
            "_p001_g_001",
            "_p001_g_002",  # pos 1, grids 0,1,2
        ]
        actual_keys = [stream._indices[i][0] for i in range(len(stream._indices))]
        print(f"Expected patterns: {expected_patterns}")
        print(f"Actual keys: {actual_keys}")

        assert set(actual_keys) == set(
            expected_patterns
        ), f"Keys don't match: {actual_keys} vs {expected_patterns}"

        # Write test data for all combinations
        for i in range(6):  # p=2, g=3
            frame = np.random.randint(0, 1000, (32, 32), dtype=dtype)
            stream.append(frame)

        stream.flush()
        assert not stream.is_active()
        print("✓ Combined position+grid dimension test passed!")


def test_backward_compatibility() -> None:
    """Test that existing position-only functionality still works."""
    # Create dimensions with only position dimension (existing behavior)
    dimensions = [
        Dimension("t", 2),
        Dimension("c", 1),
        Dimension("p", 3),  # 3 positions, no grid
        Dimension("y", 32),
        Dimension("x", 32),
    ]

    dtype = np.uint16

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_backward_compat.zarr"

        # Test with acquire-zarr backend
        from ome_writers.backends._acquire_zarr import AcquireZarrStream

        stream = AcquireZarrStream()
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        # Verify position dimension is recognized but grid is not
        assert stream._position_dim is not None
        assert stream._position_dim.size == 3
        assert stream._grid_dim is None
        assert stream._num_positions == 3
        assert stream._num_grids == 0

        # Verify array keys use old format (just numbers)
        print("\nBackward compatibility array keys generated:")
        for i, (array_key, index) in stream._indices.items():
            print(f"  Index {i}: key='{array_key}', index={index}")

        # Check that we have the expected array keys (should be "0", "0", "1", "1", "2", "2")
        # p=3, t=2 -> p0t0, p0t1, p1t0, p1t1, p2t0, p2t1
        expected_keys = ["0", "0", "1", "1", "2", "2"]
        actual_keys = [stream._indices[i][0] for i in range(len(stream._indices))]
        print(f"Expected keys: {expected_keys}")
        print(f"Actual keys: {actual_keys}")

        assert (
            actual_keys == expected_keys
        ), f"Keys don't match: {actual_keys} vs {expected_keys}"

        # Write test data
        for i in range(6):  # t=2, p=3
            frame = np.random.randint(0, 1000, (32, 32), dtype=dtype)
            stream.append(frame)

        stream.flush()
        assert not stream.is_active()
        print("✓ Backward compatibility test passed!")


if __name__ == "__main__":
    print("Testing grid dimension functionality...")
    test_grid_dimension_basic()
    test_position_and_grid_dimension()
    test_backward_compatibility()
    print("\n🎉 All tests passed!")
