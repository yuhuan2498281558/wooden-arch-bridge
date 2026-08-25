from __future__ import annotations

import unittest

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from ml_pipeline.train.partial_pooling_alpha import (
    choose_confirmed_model,
    cross_validated_joint_predictions,
    fit_joint_model,
    grouped_conformal_summary,
    predict_joint_model,
    select_joint_spec,
)


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(16):
        front = index % 2 == 0
        span = 10.0 + index
        rows.append({
            "span_m": span,
            "three_miao_rise_span_ratio": 0.18 + 0.001 * index,
            "design_target_alpha": (
                0.24 + 0.008 * span if front else 0.56 + 0.006 * span
            ),
            "sample_key": f"s{index}",
            "split_group_key": f"g{index}",
            "bridge_key": f"b{index}",
            "bridge_name": f"bridge-{index}",
        })
    return rows


class PartialPoolingAlphaTests(unittest.TestCase):
    def test_partial_pooling_interaction_is_shrunk(self) -> None:
        weak = fit_joint_model(
            _rows(),
            {
                "model_name": "partial_pooling",
                "shared_penalty": 0.0,
                "interaction_penalty": 0.0,
            },
        )
        strong = fit_joint_model(
            _rows(),
            {
                "model_name": "partial_pooling",
                "shared_penalty": 0.0,
                "interaction_penalty": 1_000_000.0,
            },
        )

        self.assertLess(
            abs(float(strong["coefficients"][3])),
            abs(float(weak["coefficients"][3])),
        )

    def test_joint_prediction_respects_explicit_mode_bounds(self) -> None:
        model = fit_joint_model(
            _rows(), {"model_name": "shared_slope", "shared_penalty": 0.1}
        )
        inputs = [
            {"span_m": 17.0},
            {"span_m": 17.0},
        ]

        prediction = predict_joint_model(model, inputs, [0, 1])

        self.assertLess(float(prediction[0]), 0.5)
        self.assertGreaterEqual(float(prediction[1]), 0.5)

    def test_logo_predictions_are_complete_and_group_isolated(self) -> None:
        prediction, audit = cross_validated_joint_predictions(
            _rows(),
            {
                "model_name": "partial_pooling",
                "shared_penalty": 0.1,
                "interaction_penalty": 10.0,
            },
            splitter=LeaveOneGroupOut(),
        )

        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(len(audit), len(_rows()))
        self.assertTrue(all(record["group_overlap"] == 0 for record in audit))

    def test_candidate_and_model_selection_prefer_simplicity_in_tolerance(self) -> None:
        selected = select_joint_spec([
            {
                "model_name": "partial_pooling",
                "shared_penalty": 0.1,
                "interaction_penalty": 10.0,
                "valid": True,
                "bridge_macro_mae": 0.050,
                "mode_macro_bridge_mae": 0.050,
                "q90": 0.10,
            },
            {
                "model_name": "partial_pooling",
                "shared_penalty": 1.0,
                "interaction_penalty": 100.0,
                "valid": True,
                "bridge_macro_mae": 0.051,
                "mode_macro_bridge_mae": 0.051,
                "q90": 0.11,
            },
        ])
        metrics = {
            name: {"mode_macro_bridge_mae": value}
            for name, value in zip(
                (
                    "mode_median",
                    "shared_slope",
                    "partial_pooling",
                    "v9_independent_experts",
                ),
                (0.052, 0.051, 0.050, 0.049),
            )
        }

        self.assertEqual(selected["interaction_penalty"], 100.0)
        self.assertEqual(choose_confirmed_model(metrics), "shared_slope")

    def test_grouped_conformal_uses_group_max_scores(self) -> None:
        rows = _rows()
        predictions = np.asarray(
            [row["design_target_alpha"] + 0.01 for row in rows], dtype=float
        )

        summary = grouped_conformal_summary(
            rows, predictions, rows, predictions, coverage=0.9
        )

        self.assertAlmostEqual(summary["radius_alpha"], 0.01)
        self.assertEqual(summary["holdout_row_coverage"], 1.0)
        self.assertEqual(summary["holdout_group_simultaneous_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
