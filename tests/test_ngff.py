import pytest
from rich import print
from yaozarrs import v05, validate_ome_object

# Helper data
COLUMN_A = {"name": "01"}
COLUMN_B = {"name": "02"}
ROW_A = {"name": "A"}
ROW_B = {"name": "B"}

WELL_A01 = {"path": "A/01", "rowIndex": 0, "columnIndex": 0}
WELL_A02 = {"path": "A/02", "rowIndex": 0, "columnIndex": 1}
WELL_B01 = {"path": "B/01", "rowIndex": 1, "columnIndex": 0}
WELL_B02 = {"path": "B/02", "rowIndex": 1, "columnIndex": 1}

ACQUISITION_1 = {
    "id": 0,
    "name": "Acquisition 1",
    "description": "First acquisition",
    "maximumfieldcount": 1,
    "starttime": 1234567890,
    "endtime": 1234567950,
}

ACQUISITION_2 = {
    "id": 1,
    "name": "Acquisition 2",
    "maximumfieldcount": 2,
}

V05_VALID_PLATES = [
    # Minimal valid plate
    {
        "version": "0.5",
        "plate": {
            "columns": [COLUMN_A, COLUMN_B],
            "rows": [ROW_A, ROW_B],
            "wells": [WELL_A01, WELL_A02, WELL_B01, WELL_B02],
        },
    },
    # # Plate with optional fields
    # {
    #     "version": "0.5",
    #     "plate": {
    #         "name": "Test Plate",
    #         "field_count": 4,
    #         "columns": [COLUMN_A, COLUMN_B],
    #         "rows": [ROW_A, ROW_B],
    #         "wells": [WELL_A01, WELL_A02, WELL_B01, WELL_B02],
    #         "acquisitions": [ACQUISITION_1, ACQUISITION_2],
    #     },
    # },
    # # Plate with single column/row
    # {
    #     "version": "0.5",
    #     "plate": {
    #         "columns": [COLUMN_A],
    #         "rows": [ROW_A],
    #         "wells": [{"path": "A/01", "rowIndex": 0, "columnIndex": 0}],
    #     },
    # },
    # # Plate with alphanumeric names
    # {
    #     "version": "0.5",
    #     "plate": {
    #         "columns": [{"name": "Col1"}, {"name": "Col2"}],
    #         "rows": [{"name": "Row1"}, {"name": "Row2"}],
    #         "wells": [
    #             {"path": "Row1/Col1", "rowIndex": 0, "columnIndex": 0},
    #             {"path": "Row1/Col2", "rowIndex": 0, "columnIndex": 1},
    #             {"path": "Row2/Col1", "rowIndex": 1, "columnIndex": 0},
    #             {"path": "Row2/Col2", "rowIndex": 1, "columnIndex": 1},
    #         ],
    #     },
    # },
]


@pytest.mark.parametrize("obj", V05_VALID_PLATES)
def test_valid_v05_plates(obj: dict) -> None:
    plate = validate_ome_object(obj, v05.Plate)
    print(plate)
    breakpoint()
