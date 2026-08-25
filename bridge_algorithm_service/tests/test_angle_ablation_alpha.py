from __future__ import annotations

import math
import unittest

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from ml_pipeline.train.angle_ablation_alpha import (
    choose_angle_candidate,
    cross_validated_feature_predictions,
    fit_shared_feature_model,
    paired_group_comparison,
    predict_shared_feature_model,
    select_penalty,
)


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(16):
        front = index % 2 == 0
        span = 10.0 + index
        rise_ratio = 0.12 + 0.006 * index
        rows.append({
            "span_m": span,
            "three_miao_rise_span_ratio": rise_ratio,
            "three_miao_design_chord_angle_deg": math.degrees(
                math.atan(3.0 * rise_ratio)
            ),
            "three_miao_rise_m": span * rise_ratio,
            "design_target_alpha": (
                0.25 + 0.006 * span if front else 0.57 + 0.004 * span
            ),
            "sample_key": f"s{index}",
            "split_group_key": f"g{index}",
            "bridge_key": f"b{index}",
            "bridge_name": f"bridge-{index}",
        })
    return rows


class AngleAblationAlphaTests(unittest.TestCase):
    def test_angle_and_rise_ratio_cannot_enter_same_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic transforms"):
            fit_shared_feature_model(
                _rows(),
                {
                    "model_name": "invalid",
                    "feature_names": [
                        "three_miao_design_chord_angle_deg",
                        "three_miao_rise_span_ratio",
                    ],
                    "penalty": 1.0,
                },
            )

    def test_angle_model_respects_explicit_mode_bounds(self) -> None:
        model = fit_shared_feature_model(
            _rows(),
            {
                "model_name": "shared_angle",
                "feature_names": ["three_miao_design_chord_angle_deg"],
                "penalty": 1.0,
            },
        )
        angle = math.degrees(math.atan(3.0 * 0.18))

        prediction = predict_shared_feature_model(
            model,
            [
                {"three_miao_design_chord_angle_deg": angle},
                {"three_miao_design_chord_angle_deg": angle},
            ],
            [0, 1],
        )

        self.assertLess(float(prediction[0]), 0.5)
        self.assertGreaterEqual(float(prediction[1]), 0.5)

    def test_logo_predictions_are_complete_and_group_isolated(self) -> None:
        prediction, audit = cross_validated_feature_predictions(
            _rows(),
            {
                "model_name": "shared_span_angle",
                "feature_names": [
                    "span_m",
                    "three_miao_design_chord_angle_deg",
                ],
                "penalty": 1.0,
            },
            splitter=LeaveOneGroupOut(),
        )

        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(len(audit), len(_rows()))
        self.assertTrue(all(record["group_overlap"] == 0 for record in audit))

    def test_penalty_selection_prefers_stronger_regularization_in_tolerance(self) -> None:
        selected = select_penalty([
            {
                "model_name": "shared_angle",
                "feature_names": ["three_miao_design_chord_angle_deg"],
                "penalty": 0.1,
                "valid": True,
                "bridge_macro_mae": 0.050,
                "mode_macro_bridge_mae": 0.050,
                "q90": 0.10,
            },
            {
                "model_name": "shared_angle",
                "feature_names": ["three_miao_design_chord_angle_deg"],
                "penalty": 10.0,
                "valid": True,
                "bridge_macro_mae": 0.051,
                "mode_macro_bridge_mae": 0.051,
                "q90": 0.11,
            },
        ])

        self.assertEqual(selected["penalty"], 10.0)

    def test_angle_candidate_prefers_single_feature_within_tolerance(self) -> None:
        metrics = {
            "shared_angle": {
                "mode_macro_bridge_mae": 0.051,
                "absolute_error_quantiles": {"q90": 0.11},
            },
            "shared_span_angle": {
                "mode_macro_bridge_mae": 0.050,
                "absolute_error_quantiles": {"q90": 0.10},
            },
        }

        self.assertEqual(choose_angle_candidate(metrics), "shared_angle")

    def test_paired_group_comparison_uses_bridge_macro_differences(self) -> None:
        rows = _rows()
        observed = np.asarray(
            [row["design_target_alpha"] for row in rows], dtype=float
        )
        baseline = observed + 0.02
        candidate = observed + 0.01

        result = paired_group_comparison(
            rows,
            candidate,
            baseline,
            bootstrap_iterations=200,
            random_seed=7,
        )

        self.assertAlmostEqual(result["mean_difference"], -0.01)
        self.assertEqual(result["candidate_better_groups"], len(rows))


if __name__ == "__main__":
    unittest.main()
