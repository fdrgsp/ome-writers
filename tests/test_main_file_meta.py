from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich import print

import ome_writers as omew

if TYPE_CHECKING:
    from pathlib import Path


def test_main_file_meta(tmp_path: Path) -> None:
    """Test multi-position TIFF metadata structure.

    For multi-position acquisitions, TifffileStream always uses main file mode:
    - First file contains complete metadata for all positions
    - Subsequent files contain BinaryOnly references
    """
    try:
        import tifffile
        from ome_types import from_xml
    except ImportError:
        pytest.skip("tifffile or ome-types is not installed")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "z": 3, "c": 2, "y": 64, "x": 64, "p": 3},
        chunk_sizes={"y": 32, "x": 32},
    )

    output_path = tmp_path / "main_file_meta.ome.tiff"

    # Create and write data (always uses main file mode for multi-position)
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()

    # Verify the metadata structure
    files = sorted(tmp_path.glob("*.ome.tiff"))
    assert len(files) == 3, "Expected 3 position files"

    # First file should have full OME metadata with all images
    with tifffile.TiffFile(files[0]) as tif:
        ome_xml = tif.ome_metadata
        ome = from_xml(ome_xml)
        print(f"\nFirst file ({files[0].name}):")
        print(ome.to_xml())
        # Should have all 3 images
        assert len(ome.images) == 3, "First file should have all 3 images"
        assert ome.binary_only is None, "First file should not have BinaryOnly"

    # Extract UUID from the first file for comparison
    first_file_uuid = None
    if ome.images and ome.images[0].pixels.tiff_data_blocks:
        tiff_data = ome.images[0].pixels.tiff_data_blocks[0]
        if tiff_data.uuid:
            first_file_uuid = tiff_data.uuid.value

    # Other files should have BinaryOnly reference
    for file in files[1:]:
        with tifffile.TiffFile(file) as tif:
            ome_xml = tif.ome_metadata
            ome = from_xml(ome_xml)
            print(f"\nSecondary file ({file.name}):")
            print(ome.to_xml())
            assert ome.binary_only is not None, f"{file.name} should have BinaryOnly"
            assert ome.binary_only.metadata_file == files[0].name
            assert ome.binary_only.uuid == first_file_uuid
            assert len(ome.images) == 0, f"{file.name} should have no images"
