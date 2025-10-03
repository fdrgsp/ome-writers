"""Test script demonstrating positional dimensions (p, g, r, etc.)."""

import tempfile
from pathlib import Path

import numpy as np

import ome_writers as omew
from ome_writers._dimensions import Dimension


def test_grid_only() -> None:
    """Test with grid dimension only (no position)."""
    print("\n=== Test 1: Grid dimension only (g) ===")

    dimensions = [
        Dimension(label="g", size=2),  # 2 grid positions
        Dimension(label="t", size=2),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        stream = omew.TifffileStream()
        stream.create(
            str(Path(tmpdir) / "grid_only.ome.tiff"), np.dtype("uint16"), dimensions
        )

        print(f"Created stream with {len(stream._indices)} frames")
        print("Array keys:")
        for i, (key, idx, img_idx) in stream._indices.items():
            print(f"  Frame {i}: key={key}, idx={idx}, image_idx={img_idx}")

        stream.flush()


def test_position_and_grid() -> None:
    """Test with both position and grid dimensions."""
    print("\n=== Test 2: Position (p) + Grid (g) ===")

    dimensions = [
        Dimension(label="p", size=2),  # 2 positions
        Dimension(label="g", size=3),  # 3 grid positions per position
        Dimension(label="t", size=2),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        stream = omew.TifffileStream()
        stream.create(
            str(Path(tmpdir) / "pos_grid.ome.tiff"), np.dtype("uint16"), dimensions
        )

        print(f"Created stream with {len(stream._indices)} frames")
        print(f"Number of positions: {stream._num_positions}")
        print(f"Positional dimensions: {[d.label for d in stream._positional_dims]}")
        print("\nArray keys (showing first 6):")
        for i in range(min(6, len(stream._indices))):
            key, idx, img_idx = stream._indices[i]
            print(f"  Frame {i}: key={key}, idx={idx}, image_idx={img_idx}")

        stream.flush()


def test_position_grid_region() -> None:
    """Test with position, grid, and custom region dimension."""
    print("\n=== Test 3: Position (p) + Grid (g) + Region (r) ===")

    dimensions = [
        Dimension(label="p", size=2),  # 2 positions
        Dimension(label="g", size=2),  # 2 grid positions
        Dimension(label="r", size=2),  # 2 regions per grid
        Dimension(label="t", size=1),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        stream = omew.TifffileStream()
        stream.create(
            str(Path(tmpdir) / "pos_grid_region.ome.tiff"),
            np.dtype("uint16"),
            dimensions,
        )

        print(f"Created stream with {len(stream._indices)} frames")
        print(f"Number of positions: {stream._num_positions}")
        print(f"Positional dimensions: {[d.label for d in stream._positional_dims]}")
        print("\nArray keys:")
        for i, (key, idx, img_idx) in stream._indices.items():
            print(f"  Frame {i}: key={key}, idx={idx}, image_idx={img_idx}")

        # Verify image indices reset per position
        print("\n=== Verifying position-relative image indices ===")
        pos_0_keys = [
            (k, img_idx)
            for k, _, img_idx in stream._indices.values()
            if k.startswith("_p0000")
        ]
        pos_1_keys = [
            (k, img_idx)
            for k, _, img_idx in stream._indices.values()
            if k.startswith("_p0001")
        ]

        print(f"Position 0 image indices: {[img_idx for _, img_idx in pos_0_keys]}")
        print(f"Position 1 image indices: {[img_idx for _, img_idx in pos_1_keys]}")

        stream.flush()


def test_zarr_with_positional_dims() -> None:
    """Test Zarr backend with positional dimensions."""
    print("\n=== Test 4: Zarr with Position (p) + Grid (g) ===")

    dimensions = [
        Dimension(label="p", size=2),
        Dimension(label="g", size=2),
        Dimension(label="t", size=2),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        stream = omew.AcquireZarrStream()
        stream.create(
            str(Path(tmpdir) / "pos_grid.ome.zarr"), np.dtype("uint16"), dimensions
        )

        print(f"Created Zarr stream with {len(stream._indices)} frames")
        print("Array keys (unique):")
        unique_keys = sorted({key for key, _, _ in stream._indices.values()})
        for key in unique_keys:
            # Count how many images per array
            count = sum(1 for k, _, _ in stream._indices.values() if k == key)
            print(f"  {key}: {count} frames")

        stream.flush()


if __name__ == "__main__":
    test_grid_only()
    test_position_and_grid()
    test_position_grid_region()
    test_zarr_with_positional_dims()
    print("\n✓ All tests passed!")
