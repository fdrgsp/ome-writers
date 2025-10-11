"""Test script demonstrating positional dimensions (p, g, r, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers._dimensions import Dimension

if TYPE_CHECKING:
    from pathlib import Path

    from .conftest import AvailableBackend


def test_tiff_grid_only(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test with grid dimension only (no position)."""
    # This test is specific to TIFF backend multi-axis support
    if backend.name != "tiff":
        pytest.skip("Grid-only multi-axis is TIFF-specific")

    dimensions = [
        Dimension(label="g", size=2),  # 2 grid positions
        Dimension(label="t", size=2),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    stream = backend.cls()
    output_path = tmp_path / f"grid_only.{backend.file_ext}"
    stream.create(
        str(output_path),
        np.dtype("uint16"),
        dimensions,
    )

    # Should have 2 grids x 2 timepoints = 4 frames
    assert len(stream._indices) == 4

    # Should have 1 "position" (no p dimension)
    assert stream._num_positions == 1

    # Should have grid as positional dimension
    assert len(stream._positional_dims) == 1
    assert stream._positional_dims[0].label == "g"

    # Check array keys are correct
    assert stream._indices[0].array_key == "_g0000"
    assert stream._indices[0].dim_index == (0, 0)  # (t=0, c=0)
    assert stream._indices[1].array_key == "_g0000"
    assert stream._indices[1].dim_index == (1, 0)  # (t=1, c=0)
    assert stream._indices[2].array_key == "_g0001"
    assert stream._indices[2].dim_index == (0, 0)
    assert stream._indices[3].array_key == "_g0001"
    assert stream._indices[3].dim_index == (1, 0)

    # Check image_id conversion
    assert stream._indices[0].image_id == "0"
    assert stream._indices[2].image_id == "1"


def test_tiff_position_and_grid(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test with both position and grid dimensions."""
    # This test is specific to TIFF backend multi-axis support
    if backend.name != "tiff":
        pytest.skip("Position + Grid multi-axis is TIFF-specific")

    dimensions = [
        Dimension(label="p", size=2),  # 2 positions
        Dimension(label="g", size=3),  # 3 grid positions per position
        Dimension(label="t", size=2),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    stream = backend.cls()
    output_path = tmp_path / f"pos_grid.{backend.file_ext}"
    stream.create(
        str(output_path),
        np.dtype("uint16"),
        dimensions,
    )

    # Should have 2 positions x 3 grids x 2 timepoints = 12 frames
    assert len(stream._indices) == 12

    # Should have 2 positions
    assert stream._num_positions == 2

    # Should have grid as positional dimension (not p)
    assert len(stream._positional_dims) == 1
    assert stream._positional_dims[0].label == "g"

    # Check first few array keys
    assert stream._indices[0].array_key == "_p0000_g0000"
    assert stream._indices[0].dim_index == (0, 0)  # (t=0, c=0)
    assert stream._indices[2].array_key == "_p0000_g0001"
    assert stream._indices[6].array_key == "_p0001_g0000"

    # Check image_id conversion for multi-axis
    assert stream._indices[0].image_id == "0:0"  # p=0, g=0
    assert stream._indices[2].image_id == "0:1"  # p=0, g=1
    assert stream._indices[4].image_id == "0:2"  # p=0, g=2
    assert stream._indices[6].image_id == "1:0"  # p=1, g=0


def test_tiff_position_grid_other(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test with position, grid, and custom other dimension."""
    # This test is specific to TIFF backend multi-axis support
    if backend.name != "tiff":
        pytest.skip("Position + Grid + Other multi-axis is TIFF-specific")

    dimensions = [
        Dimension(label="p", size=2),  # 2 positions
        Dimension(label="g", size=2),  # 2 grid positions
        Dimension(label="o", size=2),  # 2 other
        Dimension(label="t", size=1),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    stream = backend.cls()
    output_path = tmp_path / f"pos_grid_other.{backend.file_ext}"
    stream.create(
        str(output_path),
        np.dtype("uint16"),
        dimensions,
    )

    # Should have 2 positions x 2 grids x 2 other x 1 timepoint = 8 frames
    assert len(stream._indices) == 8

    # Should have 2 positions
    assert stream._num_positions == 2

    # Should have grid and other as positional dimensions
    assert len(stream._positional_dims) == 2
    assert stream._positional_dims[0].label == "g"
    assert stream._positional_dims[1].label == "o"

    # Check all array keys follow p_g_o pattern
    expected_keys = [
        "_p0000_g0000_o0000",
        "_p0000_g0000_o0001",
        "_p0000_g0001_o0000",
        "_p0000_g0001_o0001",
        "_p0001_g0000_o0000",
        "_p0001_g0000_o0001",
        "_p0001_g0001_o0000",
        "_p0001_g0001_o0001",
    ]
    for i, expected_key in enumerate(expected_keys):
        assert stream._indices[i].array_key == expected_key
        assert stream._indices[i].dim_index == (0, 0)  # (t=0, c=0)

    # Check image_id conversion for 3-axis
    assert stream._indices[0].image_id == "0:0:0"  # p=0, g=0, o=0
    assert stream._indices[1].image_id == "0:0:1"  # p=0, g=0, o=1
    assert stream._indices[2].image_id == "0:1:0"  # p=0, g=1, o=0
    assert stream._indices[4].image_id == "1:0:0"  # p=1, g=0, o=0


def test_zarr_with_positional_dims(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test Zarr backends with positional dimensions (p and g).

    Zarr backends now treat non-standard dimensions (like 'g') as positional,
    creating separate arrays for each combination of positional dimensions.
    """
    # This test is for Zarr backends (acquire-zarr and tensorstore)
    if backend.name == "tiff":
        pytest.skip("This test is for Zarr backends only")

    dimensions = [
        Dimension(label="p", size=2),
        Dimension(label="g", size=2),
        Dimension(label="t", size=2),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, chunk_size=16),
        Dimension(label="x", size=32, chunk_size=16),
    ]

    stream = backend.cls()
    output_path = tmp_path / f"pos_grid.{backend.file_ext}"
    stream.create(
        str(output_path),
        np.dtype("uint16"),
        dimensions,
    )

    # Should have 2 positions x 2 grids x 2 timepoints = 8 frames
    assert len(stream._indices) == 8

    # Should have 2 positions
    assert stream._num_positions == 2

    # Grid dimension is now treated as positional (like TIFF backend)
    assert len(stream._positional_dims) == 1
    assert stream._positional_dims[0].label == "g"

    # Non-position dims should only contain standard dimensions (t, c, y, x)
    non_pos_labels = [d.label for d in stream._non_position_dims]
    assert "g" not in non_pos_labels
    assert "t" in non_pos_labels

    # Get unique array keys
    unique_keys = sorted({array_key for array_key, _ in stream._indices.values()})

    # Should have 4 unique arrays (2 positions x 2 grids)
    assert len(unique_keys) == 4
    expected_keys = ["p0000_g0000", "p0000_g0001", "p0001_g0000", "p0001_g0001"]
    assert unique_keys == expected_keys

    # Each array should have 2 frames (2 timepoints)
    for key in unique_keys:
        count = sum(1 for array_key, _ in stream._indices.values() if array_key == key)
        assert count == 2
