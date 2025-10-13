from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

try:
    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata._ome import create_ome_metadata
except ImportError:
    pytest.skip("pymmcore_plus is not installed", allow_module_level=True)


if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1

    from .conftest import AvailableBackend


def test_pymmcore_plus_mda(tmp_path: Path, backend: AvailableBackend) -> None:
    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=0.001, loops=3),  # type: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=[(0, 0), (0.1, 0.1)],  # type: ignore
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / f"test_pymmcore_plus_mda{backend.file_ext}"
    stream = omew.create_stream(
        dest,
        dimensions=omew.dims_from_useq(
            seq, core.getImageWidth(), core.getImageHeight()
        ),
        dtype=np.uint16,
        overwrite=True,
        backend=backend.name,
    )

    @core.mda.events.frameReady.connect
    def _on_frame_ready(
        frame: np.ndarray, event: useq.MDAEvent, metadata: FrameMetaV1
    ) -> None:
        stream.append(frame)

    core.mda.run(seq)
    stream.flush()

    # make assertions
    if backend.file_ext.endswith(".tiff"):
        assert os.path.exists(str(dest).replace(".ome.tiff", "_p000.ome.tiff"))
    else:
        assert dest.exists()


class PYMMCP:
    """A little example of how one might integrate pymmcore_plus MDA with ome-writers.

    This class listens to pymmcore_plus MDA events and writes data to an OME-TIFF
    file using ome-writers. It also collects metadata during the acquisition and
    updates the OME metadata at the end of the sequence.
    """

    def __init__(
        self,
        sequence: useq.MDASequence,
        core: CMMCorePlus,
        dest: Path,
        backend: AvailableBackend,
        ome_main_file: bool = False,
    ) -> None:
        self._seq = sequence
        self._core = core
        self._dest = dest

        self._summary_meta: SummaryMetaV1 = {}  # type: ignore
        self._frame_meta_list: list[FrameMetaV1] = []

        self._stream = omew.create_stream(
            self._dest,
            dimensions=omew.dims_from_useq(
                self._seq, core.getImageWidth(), core.getImageHeight()
            ),
            dtype=np.uint16,
            overwrite=True,
            backend=backend.name,
            ome_main_file=ome_main_file,
        )

        @core.mda.events.sequenceStarted.connect
        def _on_sequence_started(
            sequence: useq.MDASequence, summary_meta: SummaryMetaV1
        ) -> None:
            self._summary_meta = summary_meta

        @core.mda.events.frameReady.connect
        def _on_frame_ready(
            frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
        ) -> None:
            self._stream.append(frame)
            self._frame_meta_list.append(frame_meta)

        @core.mda.events.sequenceFinished.connect
        def _on_sequence_finished(sequence: useq.MDASequence) -> None:
            self._stream.flush()
            if hasattr(self._stream, "update_ome_metadata"):
                ome = create_ome_metadata(self._summary_meta, self._frame_meta_list)
                self._stream.update_ome_metadata(ome)

    def run(self) -> None:
        self._core.mda.run(self._seq)


TEST_SEQ = [
    useq.MDASequence(
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=useq.WellPlatePlan(
            plate=useq.WellPlate.from_str("96-well"),
            a1_center_xy=(0, 0),
            selected_wells=((0, 0), (0, 1)),
            well_points_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
    ),
    useq.MDASequence(
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=[(0, 0), (0.1, 0.1), (0.2, 0.2)],  # type: ignore
    ),
]


@pytest.mark.parametrize("seq", TEST_SEQ)
@pytest.mark.parametrize("ome_main_file", [False, True])
def test_pymmcore_plus_mda_metadata_update(
    tmp_path: Path,
    backend: AvailableBackend,
    seq: useq.MDASequence,
    ome_main_file: bool,
) -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition.

    Tests both standard mode (ome_main_file=False) where each position file has
    its own metadata, and main file mode (ome_main_file=True) where the first
    position file contains complete metadata for all positions.
    """
    # Zarr backend tests
    if backend.file_ext.endswith(".zarr"):
        # Skip zarr for ome_main_file mode since it's TIFF-specific
        if ome_main_file:
            pytest.skip("ome_main_file mode is specific to TIFF backend")
        dest = tmp_path / "test_meta_update.ome.zarr"

    else:  # TIFF backend tests
        dest = tmp_path / "test_meta_update.ome.tiff"

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    pymm = PYMMCP(seq, core, dest, backend=backend, ome_main_file=ome_main_file)
    pymm.run()

    if backend.file_ext.endswith(".zarr"):
        assert dest.exists()
        for p in range(len(seq.stage_positions)):
            assert (dest / str(p)).exists()
        assert (dest / "zarr.json").exists()
        return

    else:  # TIFF backend tests
        try:
            import tifffile
            from ome_types import from_xml
        except ImportError:
            pytest.skip("tifffile or ome-types is not installed")
        uuid_map = {}

        # Reopen files and validate OME metadata
        for idx, f in enumerate(sorted(tmp_path.glob("*.ome.tiff"))):
            with tifffile.TiffFile(f) as tif:
                ome_xml = tif.ome_metadata
                if ome_xml is not None:
                    # Validate by attempting to parse
                    ome = from_xml(ome_xml)

                    # ome_main_file mode: non-first files have BinaryOnly
                    if ome_main_file and idx > 0:
                        assert ome.binary_only
                        assert ome.binary_only.uuid is not None
                        # BinaryOnly should always reference the first file (position 0)
                        if 0 in uuid_map:
                            assert ome.binary_only.uuid == uuid_map[0]
                    else:
                        # Standard mode or first file in ome_main_file mode
                        assert not ome.binary_only

                        # In ome_main_file mode, first file contains all images
                        # In standard mode, each file contains only one image
                        if ome_main_file and idx == 0:
                            # First file should have all metadata with all UUIDs
                            for i, img in enumerate(ome.images):
                                uuid = img.pixels.tiff_data_blocks[0].uuid
                                assert uuid is not None
                                uuid_map[i] = uuid.value
                            # Assert UUIDs are all unique
                            assert len(set(uuid_map.values())) == len(uuid_map)
                        else:
                            # Standard mode: each file has one image
                            assert len(ome.images) == 1
                            # UUID should be preserved from original file
                            if ome.images[0].pixels.tiff_data_blocks:
                                uuid = ome.images[0].pixels.tiff_data_blocks[0].uuid
                                if uuid is not None:
                                    uuid_map[idx] = uuid.value

                        # Assert there is plate information if using WellPlatePlan
                        if isinstance(seq.stage_positions, useq.WellPlatePlan):
                            assert ome.plates

                        # Verify image name (preserved from original filename)
                        if isinstance(seq.stage_positions, useq.WellPlatePlan):
                            map_names = {
                                0: f"A1_0000_p{idx:04d}",
                                1: f"A1_0001_p{idx:04d}",
                                2: f"A2_0000_p{idx:04d}",
                                3: f"A2_0001_p{idx:04d}",
                            }
                            assert ome.images[0].name == map_names[idx]
                        else:
                            assert ome.images[0].name == f"p{idx:04d}"
