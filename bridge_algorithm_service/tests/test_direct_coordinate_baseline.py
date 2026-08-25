from __future__ import annotations

import unittest

import numpy as np

from bridge_algorithm_service.node_geometry import Point
from ml_pipeline.train.direct_coordinate_baseline import (
    _decode_direct,
    _direct_targets,
    _from_local,
    _to_local,
)


def _row() -> dict[str, object]:
    row: dict[str, object] = {
        "sample_key": "bridge-a:span-01-of-01",
        "image_width_px": 1000.0,
        "image_height_px": 600.0,
        "A_x_px": 100.0,
        "A_y_px": 400.0,
        "B_x_px": 250.0,
        "B_y_px": 250.0,
        "C_x_px": 650.0,
        "C_y_px": 250.0,
        "D_x_px": 800.0,
        "D_y_px": 400.0,
    }
    for index, name in enumerate(("Q1", "Q2", "Q3", "Q4")):
        row[f"{name}_x_px"] = 180.0 + index * 170.0
        row[f"{name}_y_px"] = 320.0 - min(index, 3 - index) * 60.0
    return row


class DirectCoordinateBaselineTests(unittest.TestCase):
    def test_local_axis_round_trip(self) -> None:
        row = _row()
        source = Point(312.5, 271.25)
        u, v = _to_local(row, source)
        restored = _from_local(row, u, v)

        self.assertAlmostEqual(restored.x, source.x)
        self.assertAlmostEqual(restored.y, source.y)

    def test_all_direct_coordinate_encodings_round_trip(self) -> None:
        row = _row()
        for method in (
            "raw_pixel_direct",
            "image_normalized_direct",
            "local_axis_direct",
        ):
            encoded = _direct_targets([row], method)
            decoded = _decode_direct([row], method, encoded)
            expected = np.asarray([
                [[row[f"{name}_x_px"], row[f"{name}_y_px"]] for name in ("Q1", "Q2", "Q3", "Q4")]
            ], dtype=float)
            np.testing.assert_allclose(decoded, expected)


if __name__ == "__main__":
    unittest.main()
