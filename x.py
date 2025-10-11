"""Example"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import useq
from pymmcore_plus import CMMCorePlus

import ome_writers as omew

if TYPE_CHECKING:
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1


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
        backend: str,
    ) -> None:
        self._seq = sequence
        self._core = core
        self._dest = dest

        self._summary_meta: SummaryMetaV1 = {}  # type: ignore
        self._frame_meta_list: list[FrameMetaV1] = []

        self._stream = omew.create_stream(
            self._dest,
            dimensions=omew.dims_from_useq(
                self._seq,
                core.getImageWidth(),
                core.getImageHeight(),
            ),
            dtype=np.uint16,
            overwrite=True,
            backend=backend,
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

    def run(self) -> None:
        """Run the acquisition sequence."""
        self._core.mda.run(self._seq)


mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration()

seq = useq.MDASequence(
    time_plan=useq.TIntervalLoops(interval=0.001, loops=2),  # type: ignore
    z_plan=useq.ZRangeAround(range=2, step=1),
    channels=["DAPI", "FITC"],  # type: ignore
    # stage_positions=useq.WellPlatePlan(
    #     plate=useq.WellPlate.from_str("96-well"),
    #     a1_center_xy=(0, 0),
    #     selected_wells=((0, 0), (0, 1)),
    # ),
    stage_positions=[(0, 0), (10000, 0)],
    grid_plan=useq.GridRowsColumns(rows=2, columns=2),
)

backend = "acquire-zarr"
dest = Path("/Users/fdrgsp/Desktop/t/z.zarr")

# backend = "tiff"
# dest = Path("/Users/fdrgsp/Desktop/t/t.tiff")

pymm = PYMMCP(seq, mmc, dest, backend=backend)
pymm.run()
