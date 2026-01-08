"""Tests for multi-TIFF behavior with BinaryOnly references."""

from __future__ import annotations

from pathlib import Path

import pytest

import ome_writers as omew


def test_multiposition_binary_only_references(tmp_path: Path) -> None:
    """Test that multi-position files use BinaryOnly references correctly."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    import tifffile
    from ome_types import from_xml

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 32, "x": 32, "p": 3},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_binary_only.ome.tiff"

    # Create and write multi-position data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()

    # Verify separate files were created for each position
    base_path = Path(str(output_path).replace(".ome.tiff", ""))
    pos_files = [
        base_path.with_name(f"{base_path.name}_p{i:03d}.ome.tiff") for i in range(3)
    ]

    for pos_file in pos_files:
        assert pos_file.exists()

    # Check position 0 (main file) has complete metadata with all images
    with tifffile.TiffFile(str(pos_files[0])) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        ome_obj = from_xml(ome_xml)

        # Main file should have all 3 images
        assert len(ome_obj.images) == 3
        assert ome_obj.images[0].id == "Image:0"
        assert ome_obj.images[1].id == "Image:1"
        assert ome_obj.images[2].id == "Image:2"

        # Verify all images have proper tiff_data_blocks
        for _i, image in enumerate(ome_obj.images):
            assert image.pixels.tiff_data_blocks is not None
            assert len(image.pixels.tiff_data_blocks) > 0
            # Check UUID references
            assert image.pixels.tiff_data_blocks[0].uuid is not None

    # Check positions 1 and 2 have BinaryOnly references
    for pos_idx in [1, 2]:
        with tifffile.TiffFile(str(pos_files[pos_idx])) as tif:
            ome_xml = tif.ome_metadata
            assert ome_xml is not None
            ome_obj = from_xml(ome_xml)

            # Should have BinaryOnly element
            assert ome_obj.binary_only is not None
            assert ome_obj.binary_only.metadata_file is not None
            assert pos_files[0].name in ome_obj.binary_only.metadata_file
            assert ome_obj.binary_only.uuid is not None

            # Should not have images in BinaryOnly files
            assert len(ome_obj.images) == 0


def test_single_position_no_binary_only(tmp_path: Path) -> None:
    """Test that single position files do not use BinaryOnly references."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    import tifffile
    from ome_types import from_xml

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_single_pos.ome.tiff"

    # Create and write single-position data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Verify file was created
    assert output_path.exists()

    # Check that it has standard metadata (not BinaryOnly)
    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        ome_obj = from_xml(ome_xml)

        # Should have exactly one image
        assert len(ome_obj.images) == 1
        assert ome_obj.images[0].id == "Image:0"

        # Should NOT have BinaryOnly element
        assert ome_obj.binary_only is None


def test_update_metadata_preserves_binary_only(tmp_path: Path) -> None:
    """Test that update_metadata preserves BinaryOnly structure for positions > 0."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    import tifffile
    from ome_types import from_xml

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "c": 1, "z": 1, "y": 32, "x": 32, "p": 2},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_update_binary_only.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Create updated metadata
    from tests.test_ome_tiff_metadata_writer import create_metadata

    updated_metadata = create_metadata(
        image_name="Updated Position 0",
        channel_name="Updated Channel",
        dtype=dtype.name,
        num_images=2,
    )

    # Update metadata
    stream.update_ome_metadata(updated_metadata)

    # Verify position 0 has complete metadata
    base_path = Path(str(output_path).replace(".ome.tiff", ""))
    pos0_file = base_path.with_name(f"{base_path.name}_p000.ome.tiff")

    with tifffile.TiffFile(str(pos0_file)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        ome_obj = from_xml(ome_xml)

        # Should have all images
        assert len(ome_obj.images) == 2
        assert ome_obj.binary_only is None

    # Verify position 1 still has BinaryOnly reference
    pos1_file = base_path.with_name(f"{base_path.name}_p001.ome.tiff")

    with tifffile.TiffFile(str(pos1_file)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        ome_obj = from_xml(ome_xml)

        # Should have BinaryOnly element
        assert ome_obj.binary_only is not None
        assert pos0_file.name in ome_obj.binary_only.metadata_file

        # Should not have images
        assert len(ome_obj.images) == 0
