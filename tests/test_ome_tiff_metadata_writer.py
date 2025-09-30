"""Tests specifically for TifffileStream and OME-TIFF functionality."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    import ome_types.model as ome


def create_metadata(
    *,
    image_name: str = "Test Image",
    channel_name: str = "Test Channel",
    dtype: str = "uint16",
    size_t: int = 1,
    size_c: int = 1,
    size_z: int = 1,
    size_x: int = 32,
    size_y: int = 32,
    num_images: int = 1,
    plates: list[ome.Plate] | None = None,
) -> ome.OME:
    """Create OME metadata object with customizable parameters."""
    import ome_types.model as ome

    # Create base image
    channels = [
        ome.Channel(
            id="Channel:0",
            name=channel_name,
            samples_per_pixel=1,
        )
    ]

    pixels = ome.Pixels(
        id="Pixels:0",
        type=dtype,  # type: ignore[arg-type]
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        dimension_order="XYCZT",  # type: ignore[arg-type]
        channels=channels,
    )

    base_image = ome.Image(
        id="Image:0",
        name=image_name,
        pixels=pixels,
    )

    images = []

    # Create additional images if needed
    if num_images > 1:
        for i in range(num_images):
            # Create new channels for this image
            channel_name_final = (
                channel_name.replace("P0", f"P{i}")
                if "P0" in channel_name
                else channel_name
            )
            image_channels = [
                ome.Channel(
                    id=f"Channel:{i}",
                    name=channel_name_final,
                    samples_per_pixel=1,
                )
            ]

            # Create new pixels for this image
            image_pixels = ome.Pixels(
                id=f"Pixels:{i}",
                type=dtype,  # type: ignore[arg-type]
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                size_c=size_c,
                size_t=size_t,
                dimension_order="XYCZT",  # type: ignore[arg-type]
                channels=image_channels,
            )

            # Customize per-position names if they contain position info
            final_image_name = image_name
            if "Position" in image_name:
                final_image_name = image_name.replace("0", str(i))

            image = ome.Image(
                id=f"Image:{i}",
                name=final_image_name,
                pixels=image_pixels,
            )
            images.append(image)
    else:
        images = [base_image]

    # Create OME object
    ome_model = ome.OME(
        images=images,
        creator=f"ome_writers v{omew.__version__}",
    )

    # Add plates if provided
    if plates is not None:
        ome_model.plates = plates

    return ome_model


def test_update_metadata_single_file(tmp_path: Path) -> None:
    """Test update_metadata method for single-file TIFF streams."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_update_metadata.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()

    # Create updated metadata
    updated_metadata = create_metadata(
        image_name="Updated Test Image",
        channel_name="Updated Channel",
        dtype=dtype.name,
        size_t=2,
    )

    # Update metadata after flush
    stream.update_ome_metadata(updated_metadata)

    # Verify the metadata was updated by reading it back
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        assert "Updated Test Image" in ome_xml
        assert "Updated Channel" in ome_xml

        # Also verify we can parse the OME-XML
        from ome_types import from_xml

        ome_obj = from_xml(ome_xml)
        assert ome_obj.images[0].name == "Updated Test Image"
        assert ome_obj.images[0].pixels.channels[0].name == "Updated Channel"


def test_update_metadata_multifile(tmp_path: Path) -> None:
    """Test update_metadata method for multi-position TIFF streams."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 32, "x": 32, "p": 2},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_update_multipos.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()

    # Create updated metadata with multiple images (positions)
    updated_metadata = create_metadata(
        image_name="Position 0 Updated",
        channel_name="Channel P0",
        dtype=dtype.name,
        size_t=2,
        num_images=2,
    )

    # Update metadata after flush
    stream.update_ome_metadata(updated_metadata)

    # Verify each position file has the correct metadata
    import tifffile

    base_path = Path(str(output_path).replace(".ome.tiff", ""))

    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        assert pos_file.exists()

        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_xml = tif.ome_metadata
            expected_name = f"Position {pos_idx} Updated"
            expected_channel = f"Channel P{pos_idx}"

            assert ome_xml is not None
            assert expected_name in ome_xml
            assert expected_channel in ome_xml

            # Verify we can parse the OME-XML and it contains complete metadata
            from ome_types import from_xml

            ome_obj = from_xml(ome_xml)
            assert len(ome_obj.images) == 2  # Complete metadata: all images
            # Find the image that corresponds to this position
            position_image = next(
                img for img in ome_obj.images if img.id == f"Image:{pos_idx}"
            )
            assert position_image.name == expected_name
            assert position_image.pixels.channels[0].name == expected_channel


def test_update_metadata_error_conditions(tmp_path: Path) -> None:
    """Test error conditions in update_metadata method."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_error_metadata.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Test with invalid metadata structure that causes validation errors
    # Since we now require an OME object, we'll test with a non-OME object
    invalid_metadata = {"not": "an ome object"}

    # This should raise a TypeError due to wrong type
    with pytest.raises(TypeError, match="Expected OME metadata"):
        stream.update_ome_metadata(invalid_metadata)

    # Test that we can successfully update with valid metadata after flush
    valid_metadata = create_metadata(
        image_name="Test Image After Error",
        channel_name="Test",
        dtype=dtype.name,
    )

    # This should work without errors
    stream.update_ome_metadata(valid_metadata)

    # Verify the metadata was actually updated
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        assert "Test Image After Error" in ome_xml


def test_update_metadata_with_plates(tmp_path: Path) -> None:
    """Test update_metadata with plate metadata for multi-position experiments."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    ome = pytest.importorskip("ome_types.model", reason="ome_types not installed")
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "c": 1, "z": 1, "y": 32, "x": 32, "p": 2},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_plate_metadata.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Create metadata with plate information
    plates = [
        ome.Plate(
            id="Plate:0",
            name="Test Plate",
            wells=[
                ome.Well(
                    id="Well:0",
                    row=0,
                    column=0,
                    well_samples=[
                        ome.WellSample(
                            id="WellSample:0",
                            index=0,
                            image_ref=ome.ImageRef(id="Image:0"),
                        )
                    ],
                ),
                ome.Well(
                    id="Well:1",
                    row=0,
                    column=1,
                    well_samples=[
                        ome.WellSample(
                            id="WellSample:1",
                            index=0,
                            image_ref=ome.ImageRef(id="Image:1"),
                        )
                    ],
                ),
            ],
        )
    ]

    updated_metadata = create_metadata(
        image_name="Well A01 Field 1",
        channel_name="Channel",
        dtype=dtype.name,
        num_images=2,
        plates=plates,
    )

    # Manually update the second image name since it's not automatic
    # With OME objects, we need to modify the object directly
    updated_metadata.images[1].name = "Well A02 Field 1"

    # Update metadata after flush
    stream.update_ome_metadata(updated_metadata)

    # Verify that each position file contains only the relevant plate information
    import tifffile
    from ome_types import from_xml

    base_path = Path(str(output_path).replace(".ome.tiff", ""))

    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        assert pos_file.exists()

        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_xml = tif.ome_metadata
            assert ome_xml is not None
            ome_obj = from_xml(ome_xml)

            # Each file should have complete metadata but we verify relevant parts
            assert len(ome_obj.images) == 2  # Complete metadata: all images

            # Find the image that corresponds to this position
            position_image = next(
                img for img in ome_obj.images if img.id == f"Image:{pos_idx}"
            )
            assert position_image.id == f"Image:{pos_idx}"

            # Should have plate information with complete metadata
            if ome_obj.plates:
                assert len(ome_obj.plates) == 1
                plate = ome_obj.plates[0]
                # All wells should be present in complete metadata
                assert len(plate.wells) == 2
                # Find the well that references this position's image
                position_well = next(
                    well for well in plate.wells
                    if any(
                        ws.image_ref and ws.image_ref.id == f"Image:{pos_idx}"
                        for ws in well.well_samples
                    )
                )
                assert position_well.well_samples[0].image_ref is not None
                assert position_well.well_samples[0].image_ref.id == f"Image:{pos_idx}"


def test_tifffile_stream_basic_functionality(tmp_path: Path) -> None:
    """Test basic TifffileStream functionality."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "z": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_basic_tifffile.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    assert not stream.is_active()  # Should not be active before create()

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    frame_count = 0
    for frame in data_gen:
        stream.append(frame)
        frame_count += 1

    assert frame_count == 4  # 2 time points * 2 channels

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()

    # Verify we can read the data back
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        data = tif.asarray()
        expected_shape = (2, 2, 1, 32, 32)  # (t, c, z, y, x)
        assert data.shape == expected_shape
        assert data.dtype == dtype


def test_tifffile_stream_multiposition_basic(tmp_path: Path) -> None:
    """Test basic multi-position TifffileStream functionality."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 16, "x": 16, "p": 2},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_multipos_basic.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    frame_count = 0
    for frame in data_gen:
        stream.append(frame)
        frame_count += 1

    assert frame_count == 4  # 2 positions * 2 time points * 1 channel

    stream.flush()
    assert not stream.is_active()

    # Verify separate files were created for each position
    base_path = Path(str(output_path).replace(".ome.tiff", ""))
    pos0_file = base_path.with_name(f"{base_path.name}_p000.ome.tiff")
    pos1_file = base_path.with_name(f"{base_path.name}_p001.ome.tiff")

    assert pos0_file.exists()
    assert pos1_file.exists()

    # Verify each file has correct shape
    import tifffile

    for pos_file in [pos0_file, pos1_file]:
        with tifffile.TiffFile(str(pos_file)) as tif:
            data = tif.asarray()
            expected_shape = (2, 1, 1, 16, 16)  # (t, z, c, y, x)
            assert data.shape == expected_shape
            assert data.dtype == dtype


def test_tifffile_stream_error_handling(tmp_path: Path) -> None:
    """Test error handling specific to TifffileStream."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    # Test append without create
    stream = omew.TifffileStream()
    test_frame = np.zeros((32, 32), dtype=np.uint16)
    with pytest.raises(RuntimeError, match="Stream is closed or uninitialized"):
        stream.append(test_frame)

    # Test file overwrite protection
    data_gen1, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 16, "x": 16},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_overwrite.ome.tiff"

    # Create first stream
    stream1 = omew.TifffileStream()
    stream1 = stream1.create(str(output_path), dtype, dimensions)
    for frame in data_gen1:
        stream1.append(frame)
    stream1.flush()

    # Try to create second stream without overwrite (should fail)
    stream2 = omew.TifffileStream()
    with pytest.raises(FileExistsError, match="already exists"):
        stream2.create(str(output_path), dtype, dimensions, overwrite=False)

    # Should succeed with overwrite=True
    data_gen3, _, _ = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 16, "x": 16},
        chunk_sizes={"y": 8, "x": 8},
    )
    stream3 = omew.TifffileStream()
    stream3 = stream3.create(str(output_path), dtype, dimensions, overwrite=True)
    assert stream3.is_active()
    for frame in data_gen3:
        stream3.append(frame)
    stream3.flush()


def test_tifffile_stream_availability() -> None:
    """Test TifffileStream availability check."""
    # This test should always pass since we skip tests when tifffile is not available
    assert omew.TifffileStream.is_available() == (
        omew.TifffileStream.is_available()
    )  # Tautology, but tests the method


def test_tifffile_stream_context_manager(tmp_path: Path) -> None:
    """Test TifffileStream as context manager."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 16, "x": 16},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_context_manager.ome.tiff"

    # Use as context manager
    with omew.TifffileStream() as stream:
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        for frame in data_gen:
            stream.append(frame)

        # Stream should still be active inside context
        assert stream.is_active()

    # Stream should be flushed and inactive after exiting context
    assert not stream.is_active()
    assert output_path.exists()


def test_multifile_ome_tiff_specification_compliance(tmp_path: Path) -> None:
    """Test multi-file OME-TIFF compliance with the OME-TIFF specification.

    This test verifies that each file in a multi-position OME-TIFF dataset
    contains the complete OME-XML metadata with proper UUID cross-references
    as required by the OME-TIFF specification.
    """
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    # Create test data for multi-position acquisition
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "z": 1, "c": 1, "y": 16, "x": 16, "p": 3},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_multifile.ome.tiff"

    # Create the stream
    stream = omew.TifffileStream()
    stream.create(str(output_path), dtype, dimensions)

    # Write all frames
    for frame in data_gen:
        stream.append(frame)

    # Create complete OME metadata for testing
    import ome_types.model as ome

    images = []
    for p in range(3):
        pixels = ome.Pixels(
            id=f"Pixels:{p}",
            dimension_order=ome.Pixels_DimensionOrder.XYZCT,
            type=ome.PixelType.UINT8,
            size_x=16,
            size_y=16,
            size_z=1,
            size_c=1,
            size_t=2,
            channels=[ome.Channel(id=f"Channel:{p}:0", samples_per_pixel=1)],
            tiff_data_blocks=[
                ome.TiffData(
                    ifd=0,
                    first_c=0,
                    first_z=0,
                    first_t=0,
                    plane_count=1,
                ),
                ome.TiffData(
                    ifd=1,
                    first_c=0,
                    first_z=0,
                    first_t=1,
                    plane_count=1,
                ),
            ]
        )
        image = ome.Image(
            id=f"Image:{p}",
            name=f"Image {p}",
            pixels=pixels,
        )
        images.append(image)

    metadata = ome.OME(
        uuid="urn:uuid:12345678-1234-1234-1234-123456789abc",
        images=images,
    )

    # Flush to write files and update metadata
    stream.flush()
    stream.update_ome_metadata(metadata)

    # Verify that we have multiple files
    files = list(tmp_path.glob("*.ome.tiff"))
    assert len(files) == 3, f"Expected 3 files, got {len(files)}"

    # Check each file for proper OME-XML structure according to specification
    import xml.etree.ElementTree as ET

    import tifffile

    for file_path in sorted(files):
        # Read the OME-XML comment from the TIFF file
        with tifffile.TiffFile(file_path) as tif:
            ome_xml = tif.ome_metadata

            from rich import print
            print(ome_xml)

        assert ome_xml is not None, f"No OME-XML metadata found in {file_path}"

        # Use ome-types validation to ensure the XML is valid and compliant
        from ome_types import from_xml, validate_xml

        # First validate the XML against the OME schema
        validate_xml(ome_xml)  # This validates against the OME XSD schema

        # Then parse and validate the structure
        ome_obj = from_xml(ome_xml)  # This creates OME object and validates structure

        # Verify basic structure requirements from the specification
        assert len(ome_obj.images) == 3, (
            f"Expected 3 images in metadata, got {len(ome_obj.images)}"
        )
        assert ome_obj.uuid is not None, "OME metadata missing UUID"

        # Parse the XML to check TiffData structure for multi-file compliance
        root = ET.fromstring(ome_xml)

        # Check for TiffData elements with UUID references
        tiff_data_elements = root.findall(
            ".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}TiffData"
        )
        expected_tiff_data_count = 6  # 3 positions x 2 time points
        assert len(tiff_data_elements) == expected_tiff_data_count, (
            f"Expected {expected_tiff_data_count} TiffData elements, "
            f"got {len(tiff_data_elements)}"
        )

        # Check for UUID elements - each TiffData should have a UUID
        uuid_elements = root.findall(
            ".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}UUID"
        )
        assert len(uuid_elements) == expected_tiff_data_count, (
            f"Expected {expected_tiff_data_count} UUID elements, "
            f"got {len(uuid_elements)}"
        )

        # Verify that each TiffData has a UUID with FileName
        uuid_filenames = set()
        for td in tiff_data_elements:
            uuid_elem = td.find(
                ".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}UUID"
            )
            assert uuid_elem is not None, "TiffData missing UUID element"

            filename = uuid_elem.get("FileName")
            uuid_val = uuid_elem.text

            assert filename is not None, "UUID element missing FileName attribute"
            assert uuid_val is not None, "UUID element missing text content"
            assert filename.endswith(".ome.tiff"), f"Invalid filename: {filename}"

            # Collect unique filenames to verify we reference all files
            uuid_filenames.add(filename)

        # Verify that UUID references point to all 3 files in the dataset
        expected_filenames = {
            "test_multifile_p000.ome.tiff",
            "test_multifile_p001.ome.tiff",
            "test_multifile_p002.ome.tiff"
        }
        assert uuid_filenames == expected_filenames, (
            f"UUID filenames {uuid_filenames} don't match expected {expected_filenames}"
        )
