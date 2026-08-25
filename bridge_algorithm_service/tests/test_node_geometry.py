from __future__ import annotations

import unittest

from bridge_algorithm_service.node_geometry import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    Point,
    geometry_payload,
    measure_ratios,
    ratios_from_result,
    reconstruct_nodes,
)


class NodeGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Point(0.0, 0.0)
        self.b = Point(3.0, 3.0)
        self.c = Point(6.0, 3.0)
        self.d = Point(9.0, 0.0)

    def test_default_reconstruction_matches_existing_rule(self) -> None:
        nodes = reconstruct_nodes(self.a, self.b, self.c, self.d)

        self.assertAlmostEqual(nodes["Q1"].x, 2.0)
        self.assertAlmostEqual(nodes["Q1"].y, 2.0)
        self.assertAlmostEqual(nodes["Q2"].x, 3.75)
        self.assertAlmostEqual(nodes["Q2"].y, 3.0)
        self.assertAlmostEqual(nodes["Q3"].x, 5.25)
        self.assertAlmostEqual(nodes["Q4"].x, 7.0)
        self.assertAlmostEqual(nodes["Q4"].y, 2.0)

    def test_measurement_round_trips_reconstructed_nodes(self) -> None:
        nodes = reconstruct_nodes(self.a, self.b, self.c, self.d, alpha=0.61, beta=0.22)
        measured = measure_ratios(self.a, self.b, self.c, self.d, **{key.lower(): value for key, value in nodes.items()})

        self.assertAlmostEqual(measured.alpha, 0.61)
        self.assertAlmostEqual(measured.beta, 0.22)
        self.assertAlmostEqual(measured.alpha_symmetry_error, 0.0)
        self.assertAlmostEqual(measured.beta_symmetry_error, 0.0)
        self.assertAlmostEqual(measured.max_normalized_projection_error, 0.0)
        self.assertEqual(measured.warnings, ())

    def test_measurement_averages_left_and_right(self) -> None:
        left = reconstruct_nodes(self.a, self.b, self.c, self.d, alpha=0.6, beta=0.2)
        right = reconstruct_nodes(self.a, self.b, self.c, self.d, alpha=0.7, beta=0.3)
        measured = measure_ratios(
            self.a,
            self.b,
            self.c,
            self.d,
            left["Q1"],
            left["Q2"],
            right["Q3"],
            right["Q4"],
        )

        self.assertAlmostEqual(measured.alpha, 0.65)
        self.assertAlmostEqual(measured.beta, 0.25)
        self.assertAlmostEqual(measured.alpha_symmetry_error, 0.1)
        self.assertAlmostEqual(measured.beta_symmetry_error, 0.1)

    def test_parallel_chord_preprocessing_separates_raised_start(self) -> None:
        base = reconstruct_nodes(self.a, self.b, self.c, self.d, alpha=0.6, beta=0.2)
        q1 = Point(base["Q1"].x, base["Q1"].y - 0.2)
        q4 = Point(base["Q4"].x, base["Q4"].y - 0.2)
        measured = measure_ratios(
            self.a, self.b, self.c, self.d,
            q1, base["Q2"], base["Q3"], q4,
        )

        self.assertAlmostEqual(measured.alpha_left, 0.6)
        self.assertAlmostEqual(measured.alpha_right, 0.6)
        self.assertAlmostEqual(measured.alpha, 0.6)
        self.assertAlmostEqual(measured.outer_start_offset_left_normalized, 0.2 / 9.0)
        self.assertAlmostEqual(measured.outer_start_offset_right_normalized, 0.2 / 9.0)
        self.assertNotAlmostEqual(measured.alpha_legacy_projection_left, 0.6)

    def test_reconstruction_rejects_invalid_ratios(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            reconstruct_nodes(self.a, self.b, self.c, self.d, alpha=1.2, beta=0.2)
        with self.assertRaisesRegex(ValueError, "beta"):
            reconstruct_nodes(self.a, self.b, self.c, self.d, alpha=0.6, beta=0.6)

    def test_measurement_rejects_degenerate_anchor_segment(self) -> None:
        nodes = reconstruct_nodes(self.a, self.b, self.c, self.d)
        with self.assertRaisesRegex(ValueError, "zero-length"):
            measure_ratios(self.a, self.a, self.c, self.d, **{key.lower(): value for key, value in nodes.items()})

    def test_payload_and_result_resolution(self) -> None:
        payload = geometry_payload(0.62, 0.23, source="calibrated_constant")
        alpha, beta, source = ratios_from_result({"geometry": payload})

        self.assertAlmostEqual(alpha, 0.62)
        self.assertAlmostEqual(beta, 0.23)
        self.assertEqual(source, "calibrated_constant")

    def test_invalid_result_falls_back_to_legacy_rule(self) -> None:
        result = {
            "geometry": {
                "five_miao_bullhead": {
                    "ratios": {"alpha_outer": "bad", "beta_inner": 0.9},
                    "source": "ml",
                }
            }
        }

        self.assertEqual(ratios_from_result(result), (DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"))


if __name__ == "__main__":
    unittest.main()
