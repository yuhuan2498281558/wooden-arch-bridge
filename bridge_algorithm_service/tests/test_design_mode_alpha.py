from __future__ import annotations

import unittest

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from ml_pipeline.train.design_mode_alpha import (
    clip_to_design_mode,
    cross_validated_predictions,
    fit_design_mode_model,
    predict_design_mode_model,
)


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(12):
        front = index % 2 == 0
        rows.append({
            "span_m": 10.0 + index,
            "three_miao_rise_span_ratio": 0.18 + 0.002 * index,
            "design_target_alpha": (0.30 + 0.01 * index) if front else (0.62 + 0.005 * index),
            "observed_alpha_left": (0.29 + 0.01 * index) if front else (0.61 + 0.005 * index),
            "observed_alpha_right": (0.31 + 0.01 * index) if front else (0.63 + 0.005 * index),
            "sample_key": f"s{index}",
            "split_group_key": f"g{index}",
            "bridge_key": f"b{index}",
            "bridge_name": f"bridge-{index}",
        })
    return rows


def _median_specs() -> dict[str, dict]:
    spec = {
        "model_name": "median",
        "feature_set": "none",
        "feature_names": [],
        "params": {},
    }
    return {"front_half": dict(spec), "back_half": dict(spec)}


class DesignModeAlphaTests(unittest.TestCase):
    def test_clip_keeps_predictions_inside_selected_half(self) -> None:
        clipped = clip_to_design_mode(
            np.asarray([-1.0, 0.8, 0.2, 2.0]),
            np.asarray([0, 0, 1, 1]),
        )

        self.assertGreater(float(clipped[0]), 0.0)
        self.assertLess(float(clipped[1]), 0.5)
        self.assertGreaterEqual(float(clipped[2]), 0.5)
        self.assertLess(float(clipped[3]), 1.0)

    def test_explicit_mode_routes_same_input_to_independent_experts(self) -> None:
        model = fit_design_mode_model(_rows(), _median_specs())
        inputs = [
            {"span_m": 15.0, "three_miao_rise_span_ratio": 0.19},
            {"span_m": 15.0, "three_miao_rise_span_ratio": 0.19},
        ]

        prediction = predict_design_mode_model(
            model,
            inputs,
            ["front_half", "back_half"],
        )

        self.assertLess(float(prediction[0]), 0.5)
        self.assertGreaterEqual(float(prediction[1]), 0.5)
        self.assertNotAlmostEqual(float(prediction[0]), float(prediction[1]))

    def test_prediction_rejects_missing_or_invalid_mode(self) -> None:
        model = fit_design_mode_model(_rows(), _median_specs())
        inputs = [{"span_m": 15.0, "three_miao_rise_span_ratio": 0.19}]

        with self.assertRaisesRegex(ValueError, "align"):
            predict_design_mode_model(model, inputs, [])
        with self.assertRaisesRegex(ValueError, "front_half/back_half"):
            predict_design_mode_model(model, inputs, ["automatic"])

    def test_logo_predictions_are_complete_and_group_isolated(self) -> None:
        prediction, fold_audit = cross_validated_predictions(
            _rows(),
            _median_specs(),
            splitter=LeaveOneGroupOut(),
        )

        self.assertEqual(len(prediction), len(_rows()))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(len(fold_audit), len(_rows()))
        self.assertTrue(all(record["group_overlap"] == 0 for record in fold_audit))
        classes = np.asarray([index % 2 for index in range(len(_rows()))])
        self.assertTrue(np.all(prediction[classes == 0] < 0.5))
        self.assertTrue(np.all(prediction[classes == 1] >= 0.5))


if __name__ == "__main__":
    unittest.main()
