from __future__ import annotations

import unittest

import numpy as np

from ml_pipeline.train.structure_gated_alpha import (
    STRUCTURE_BOUNDARY_BAND,
    UNCOMMITTED_STRUCTURE_CLASS,
    UNCOMMITTED_STRUCTURE_MODE,
    alpha_structure_class,
    derived_structure_class,
    derived_structure_mode,
    fit_structure_gated_model,
    in_structure_boundary_band,
    left_right_structure_disagreement,
    predict_structure_gated_model,
)


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index, (span, alpha) in enumerate([
        (10.0, 0.32),
        (11.0, 0.36),
        (12.0, 0.40),
        (24.0, 0.60),
        (25.0, 0.64),
        (26.0, 0.68),
    ]):
        rows.append({
            "span_m": span,
            "three_miao_rise_span_ratio": 0.18,
            "design_target_alpha": alpha,
            "split_group_key": f"g{index}",
            "bridge_key": f"b{index}",
        })
    return rows


class StructureGatedAlphaTests(unittest.TestCase):
    def test_structure_threshold_is_exactly_half(self) -> None:
        # Geometric clip bound for a committed designer-selected half.
        self.assertEqual(alpha_structure_class(0.499999), 0)
        self.assertEqual(alpha_structure_class(0.5), 1)

    def test_yuanji_like_mean_is_not_committed_to_back_half(self) -> None:
        # 远济 holdout: mean 0.503, L/R 0.506/0.500. Knife-edge 0.5 was a false split.
        self.assertEqual(
            derived_structure_mode(0.503, 0.506, 0.500),
            UNCOMMITTED_STRUCTURE_MODE,
        )
        self.assertEqual(derived_structure_class(0.503, 0.506, 0.500), UNCOMMITTED_STRUCTURE_CLASS)
        self.assertTrue(in_structure_boundary_band(0.503))
        self.assertFalse(left_right_structure_disagreement(0.506, 0.500))

    def test_clear_back_and_front_samples_stay_committed(self) -> None:
        self.assertEqual(derived_structure_mode(0.745, 0.75, 0.74), "back_half")
        self.assertEqual(derived_structure_class(0.745, 0.75, 0.74), 1)
        self.assertEqual(derived_structure_mode(0.21, 0.26, 0.16), "front_half")
        self.assertEqual(derived_structure_class(0.21, 0.26, 0.16), 0)

    def test_tiny_left_right_straddle_is_not_structural_disagreement(self) -> None:
        self.assertFalse(left_right_structure_disagreement(0.506, 0.4997))
        self.assertFalse(left_right_structure_disagreement(0.503, 0.497))
        self.assertEqual(STRUCTURE_BOUNDARY_BAND, 0.03)
        # Both sides clearly opposite halves still counts as mixed construction.
        self.assertTrue(left_right_structure_disagreement(0.26, 0.74))

    def test_mean_only_path_uses_the_same_boundary_band(self) -> None:
        self.assertEqual(derived_structure_mode(0.503), UNCOMMITTED_STRUCTURE_MODE)
        self.assertEqual(derived_structure_mode(0.26), "front_half")
        self.assertEqual(derived_structure_mode(0.75), "back_half")

    def test_gate_routes_to_two_independent_experts(self) -> None:
        model = fit_structure_gated_model(
            _rows(),
            gate_feature_set="span_only",
            expert_feature_set="span_only",
            gate_c=1.0,
            expert_model="median",
            expert_ridge_alpha=None,
        )

        prediction, probability, predicted_class = predict_structure_gated_model(
            model,
            [
                {"span_m": 10.5, "three_miao_rise_span_ratio": 0.18},
                {"span_m": 25.5, "three_miao_rise_span_ratio": 0.18},
            ],
        )

        self.assertTrue(np.allclose(prediction, [0.36, 0.64]))
        self.assertEqual(predicted_class.tolist(), [0, 1])
        self.assertLess(float(probability[0]), 0.5)
        self.assertGreaterEqual(float(probability[1]), 0.5)

    def test_oracle_diagnostic_does_not_change_gate_probability(self) -> None:
        model = fit_structure_gated_model(
            _rows(),
            gate_feature_set="span_only",
            expert_feature_set="span_only",
            gate_c=1.0,
            expert_model="median",
            expert_ridge_alpha=None,
        )
        test_rows = [{"span_m": 10.5, "three_miao_rise_span_ratio": 0.18}]

        automatic, automatic_probability, _ = predict_structure_gated_model(model, test_rows)
        oracle, oracle_probability, _ = predict_structure_gated_model(
            model, test_rows, oracle_classes=np.asarray([1])
        )

        self.assertAlmostEqual(float(automatic_probability[0]), float(oracle_probability[0]))
        self.assertLess(float(automatic[0]), 0.5)
        self.assertGreaterEqual(float(oracle[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
