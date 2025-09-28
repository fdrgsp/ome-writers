"""High Content Screening (HCS) metadata classes for OME-NGFF 0.5 support."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import model_validator

from ome_writers.model import FrozenBaseModel


class AcquisitionNGFF(FrozenBaseModel):
    """A single acquisition in an HCS plate.

    Represents an acquisition event that can contain multiple fields of view.
    Each acquisition has a unique ID within the context of the plate.
    """

    id: int
    name: str | None = None
    maximumfieldcount: int | None = None
    description: str | None = None
    starttime: int | None = None  # epoch timestamp
    endtime: int | None = None  # epoch timestamp


class RowNGFF(FrozenBaseModel):
    """A single row in an HCS plate.

    Row names must be alphanumeric and unique within the plate.
    """

    name: str


class ColumnNGFF(FrozenBaseModel):
    """A single column in an HCS plate.

    Column names must be alphanumeric and unique within the plate.
    """

    name: str


class WellInPlateNGFF(FrozenBaseModel):
    """A single well within an HCS plate.

    The path must be in the format "{row_name}/{column_name}".
    Row and column indices are 0-based.
    """

    path: str
    rowIndex: int
    columnIndex: int


class WellImageNGFF(FrozenBaseModel):
    """A single image (field of view) within a well.

    Each image has a path and optionally references an acquisition.
    """

    path: str
    acquisition: int | None = None


class WellNGFF(FrozenBaseModel):
    """Metadata for a single well in an HCS plate.

    Contains all images (fields of view) within the well.
    """

    images: Sequence[WellImageNGFF]
    version: str = "0.5"


class PlateNGFF(FrozenBaseModel):
    """A single HCS plate containing wells arranged in rows and columns.

    Follows the OME-NGFF 0.5 specification for high-content screening.
    """

    columns: Sequence[ColumnNGFF]
    rows: Sequence[RowNGFF]
    wells: Sequence[WellInPlateNGFF]
    acquisitions: Sequence[AcquisitionNGFF] | None = None
    field_count: int | None = None
    name: str | None = None
    version: str = "0.5"

    @model_validator(mode="after")
    def validate_plate(self) -> PlateNGFF:
        """Validate that all well paths reference existing rows and columns."""
        self._validate_well_paths()
        self._validate_unique_names()
        return self

    def _validate_well_paths(self) -> None:
        """Validate that all well paths reference existing rows and columns."""
        row_names = {row.name for row in self.rows}
        column_names = {column.name for column in self.columns}

        errors = []
        for well in self.wells:
            path = well.path
            if path.count("/") != 1:
                errors.append(f"well path '{path}' does not contain a single '/'")
                continue

            row, column = path.split("/")
            if row not in row_names:
                errors.append(
                    f"row '{row}' in well path '{path}' is not in list of rows"
                )
            if column not in column_names:
                errors.append(
                    f"column '{column}' in well path '{path}' is not in list of columns"
                )

        if errors:
            raise ValueError("Invalid well paths:\n" + "\n".join(errors))

    def _validate_unique_names(self) -> None:
        """Validate that row and column names are unique."""
        row_names = [row.name for row in self.rows]
        if len(row_names) != len(set(row_names)):
            raise ValueError("Row names must be unique")

        column_names = [column.name for column in self.columns]
        if len(column_names) != len(set(column_names)):
            raise ValueError("Column names must be unique")
