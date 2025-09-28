from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    def __init__(
        self, sequence: useq.MDASequence, core: CMMCorePlus, dest: Path, backend: str
    ) -> None:
        self._seq = sequence
        self._core = core
        self._dest = dest
        self._backend = backend

        self._stream: omew.OMEStream

        self._summary_meta: SummaryMetaV1 = {}  # type: ignore
        self._frame_meta_list: list[FrameMetaV1] = []

        @core.mda.events.sequenceStarted.connect
        def _on_sequence_started(
            sequence: useq.MDASequence, summary_meta: SummaryMetaV1
        ) -> None:
            plate, wells = omew.ngff_plate_and_wells_from_useq(self._seq)
            self._stream = omew.create_stream(
                self._dest,
                dimensions=omew.dims_from_useq(
                    self._seq, core.getImageWidth(), core.getImageHeight()
                ),
                plate=plate,
                wells=wells,
                dtype=np.uint16,
                overwrite=True,
                backend=cast("omew.BackendName", self._backend),
            )
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
            _o = create_ome_metadata(self._summary_meta, self._frame_meta_list)
            # self._stream.update_metadata(dict(_o))

    def run(self) -> None:
        self._core.mda.run(self._seq)


# def test_pymmcore_plus_mda_acquire_zarr(tmp_path: Path) -> None:
def test_pymmcore_plus_mda_acquire_zarr() -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition."""

    # skip if acquire_zarr is not installed
    try:
        import acquire_zarr  # noqa: F401
    except ImportError:
        pytest.skip("acquire_zarr is not installed", allow_module_level=True)

    seq = useq.MDASequence(
        # time_plan=useq.TIntervalLoops(interval=0.001, loops=2),  # type: ignore
        # z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=useq.WellPlatePlan(
            plate=useq.WellPlate.from_str("96-well"),
            a1_center_xy=(0, 0),
            selected_wells=((0, 0), (0, 1)),
        ),
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = Path("/Users/fdrgsp/Desktop/t/test_acq_zarr.zarr")

    pymm = PYMMCP(seq, core, dest, backend="acquire-zarr")
    pymm.run()

    json_attr = dest / "zarr.json"
    assert json_attr.exists()
    with open(json_attr) as f:
        import json

        from rich import print

        d = json.load(f)
        print(d)
