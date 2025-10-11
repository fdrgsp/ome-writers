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
        assert os.path.exists(str(dest).replace(".ome.tiff", "_p0000.ome.tiff"))
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
        time_plan=useq.TIntervalLoops(interval=0.001, loops=2),  # type: ignore
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
        stage_positions=[(0, 0), (0.1, 0.1)],  # type: ignore
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),  # type: ignore
    ),
]


@pytest.mark.parametrize("seq", TEST_SEQ)
def test_pymmcore_plus_mda_metadata_update(
    tmp_path: Path, backend: AvailableBackend, seq: useq.MDASequence
) -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition."""

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / f"test_mda_tiff_metadata_update.{backend.file_ext}"

    pymm = PYMMCP(seq, core, dest, backend=backend)
    pymm.run()

    if backend.file_ext.endswith("tiff"):
        # TO pass requires:
        # https://github.com/pymmcore-plus/pymmcore-plus/pull/505
        # https://github.com/pymmcore-plus/pymmcore-plus/pull/510
        try:
            import tifffile
            from ome_types import from_xml
        except ImportError:
            pytest.skip("tifffile or ome-types is not installed")
        # reopen the file and validate ome metadata
        for f in list(tmp_path.glob("*.ome.tiff")):
            with tifffile.TiffFile(f) as tif:
                ome_xml = tif.ome_metadata
                if ome_xml is not None:
                    # validate by attempting to parse
                    ome = from_xml(ome_xml)
                    # assert there is plate information
                    if isinstance(seq.stage_positions, useq.WellPlatePlan):
                        assert ome.plates

    elif backend.file_ext.endswith("zarr"):
        assert dest.exists()
        # With the new descriptive naming, folders are named like "p0000_g0000"
        # Calculate expected arrays: num_positions * grid_positions
        num_positions = len(seq.stage_positions)
        g = seq.grid_plan.num_positions() if seq.grid_plan else 1

        # Generate expected folder names
        expected_folders = []
        for p in range(num_positions):
            for grid in range(g):
                if g > 1:
                    # Multiple grid positions: use "p{:04d}_g{:04d}" format
                    expected_folders.append(f"p{p:04d}_g{grid:04d}")
                else:
                    # Single grid position: use "p{:04d}" format
                    expected_folders.append(f"p{p:04d}")

        # Verify all expected folders exist
        for folder_name in expected_folders:
            assert (dest / folder_name).exists(), f"Missing folder: {folder_name}"

        assert (dest / "zarr.json").exists()
