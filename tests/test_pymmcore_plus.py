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
def test_pymmcore_plus_mda_metadata_update(
    tmp_path: Path, backend: AvailableBackend, seq: useq.MDASequence
) -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition."""
    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / "test_meta_update.ome.tiff"

    pymm = PYMMCP(seq, core, dest, backend=backend)
    pymm.run()

    if backend.file_ext.endswith("tiff"):
        try:
            import tifffile
            from ome_types import from_xml
        except ImportError:
            pytest.skip("tifffile or ome-types is not installed")
        uuid_map = {}
        # reopen the file and validate ome metadata
        for idx, f in enumerate(sorted(tmp_path.glob("*.ome.tiff"))):
            with tifffile.TiffFile(f) as tif:
                ome_xml = tif.ome_metadata
                if ome_xml is not None:
                    # validate by attempting to parse
                    ome = from_xml(ome_xml)
                    # assert there is plate information
                    if isinstance(seq.stage_positions, useq.WellPlatePlan):
                        assert ome.plates
                    # verify tiff_data_blocks uuid
                    assert len(ome.images) == 1
                    uuid_map[idx] = ome.images[0].pixels.tiff_data_blocks[0].uuid
                    # verify image name
                    if isinstance(seq.stage_positions, useq.WellPlatePlan):
                        if idx == 0:
                            well_name = f"A1_0000_p{idx:04d}"
                        elif idx == 1:
                            well_name = f"A1_0001_p{idx:04d}"
                        elif idx == 2:
                            well_name = f"A2_0000_p{idx:04d}"
                        else:
                            well_name = f"A2_0001_p{idx:04d}"
                    else:
                        well_name = f"p{idx:04d}"
                    assert ome.images[0].name == well_name
        # assert uuids are all unique
        assert len(uuid_map) == len(seq.stage_positions)

    elif backend.file_ext.endswith("zarr"):
        assert dest.exists()
        for p in range(len(seq.stage_positions)):
            assert (dest / str(p)).exists()
        assert (dest / "zarr.json").exists()


@pytest.mark.parametrize("seq", TEST_SEQ)
def test_pymmcore_plus_mda_tiff_metadata_main_file_meta_update(
    tmp_path: Path, backend: AvailableBackend, seq: useq.MDASequence
) -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition."""
    if backend.file_ext.endswith("zarr"):
        pytest.skip("This test is specific to TIFF backend")

    # skip if tifffile or ome-types is not installed
    try:
        import tifffile
        from ome_types import from_xml
    except ImportError:
        pytest.skip("tifffile or ome-types is not installed")

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / "test_main_file_meta_update.ome.tiff"

    pymm = PYMMCP(seq, core, dest, backend=backend, ome_main_file=True)
    pymm.run()

    uuid_map = {}

    # reopen the file and validate ome metadata
    for idx, f in enumerate(sorted(tmp_path.glob("*.ome.tiff"))):
        with tifffile.TiffFile(f) as tif:
            ome_xml = tif.ome_metadata
            if ome_xml is not None:
                # validate by attempting to parse
                ome = from_xml(ome_xml)
                # assert there is binary_only information
                if idx > 0:
                    assert ome.binary_only
                    assert ome.binary_only.uuid is not None
                    # BinaryOnly should always reference the first file (position 0)
                    assert ome.binary_only.uuid == uuid_map[0]
                else:
                    assert not ome.binary_only
                    # assert there is plate information
                    if isinstance(seq.stage_positions, useq.WellPlatePlan):
                        assert ome.plates
                    # this is the first file, should have all metadata with all uuids
                    for i, img in enumerate(ome.images):
                        # verify image name
                        if isinstance(seq.stage_positions, useq.WellPlatePlan):
                            if idx == 0:
                                well_name = f"A1_0000_p{idx:04d}"
                            elif idx == 1:
                                well_name = f"A1_0001_p{idx:04d}"
                            elif idx == 2:
                                well_name = f"A2_0000_p{idx:04d}"
                            else:
                                well_name = f"A2_0001_p{idx:04d}"
                        else:
                            well_name = f"p{idx:04d}"
                        assert ome.images[0].name == well_name
                        # verify tiff_data_blocks uuid
                        uuid = img.pixels.tiff_data_blocks[0].uuid
                        assert uuid is not None
                        uuid_map[i] = uuid.value
                    # assert uuids are all unique
                    assert len(set(uuid_map.values())) == len(uuid_map)
