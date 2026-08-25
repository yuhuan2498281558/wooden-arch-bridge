from __future__ import annotations

import unittest

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from ml_pipeline.train.design_mode_alpha import (
    _classes,
    _row_flags,
    clip_to_design_mode,
    cross_validated_predictions,
    fit_design_mode_model,
    predict_design_mode_model,
)
from ml_pipeline.train.structure_gated_alpha import UNCOMMITTED_STRUCTURE_MODE


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


def _yuanji_row() -> dict:
    return {
        "span_m": 16.0,
        "three_miao_rise_span_ratio": 0.19,
        "design_target_alpha": 0.503,
        "observed_alpha_left": 0.506,
        "observed_alpha_right": 0.500,
        "sample_key": "yuanji",
        "split_group_key": "g-yuanji",
        "bridge_key": "b-yuanji",
        "bridge_name": "远济桥",
    }


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

    def test_clip_uncommitted_does_not_force_a_half(self) -> None:
        clipped = clip_to_design_mode(np.asarray([0.503, 0.42]), np.asarray([-1, -1]))
        self.assertAlmostEqual(float(clipped[0]), 0.503)
        self.assertAlmostEqual(float(clipped[1]), 0.42)

    def test_yuanji_like_is_uncommitted_and_not_a_half_disagreement(self) -> None:
        yuanji = _yuanji_row()
        self.assertEqual(int(_classes([yuanji])[0]), -1)
        boundary, disagreement = _row_flags(yuanji)
        self.assertTrue(boundary)
        self.assertFalse(disagreement)

        clear_back = {
            **yuanji,
            "design_target_alpha": 0.745,
            "observed_alpha_left": 0.75,
            "observed_alpha_right": 0.74,
        }
        clear_front = {
            **yuanji,
            "design_target_alpha": 0.21,
            "observed_alpha_left": 0.26,
            "observed_alpha_right": 0.16,
        }
        self.assertEqual(int(_classes([clear_back])[0]), 1)
        self.assertEqual(int(_classes([clear_front])[0]), 0)
        self.assertFalse(_row_flags(clear_back)[1])
        self.assertFalse(_row_flags(clear_front)[1])
        self.assertFalse(_row_flags({
            **yuanji,
            "observed_alpha_left": 0.506,
            "observed_alpha_right": 0.4997,
        })[1])

    def test_uncommitted_row_is_not_trained_as_back_half(self) -> None:
        rows = _rows()
        rows.append(_yuanji_row())
        model = fit_design_mode_model(rows, _median_specs())
        back_targets = [
            (0.62 + 0.005 * index) for index in range(12) if index % 2 == 1
        ]
        self.assertAlmostEqual(
            float(model["experts"]["back_half"]["constant"]),
            float(np.median(back_targets)),
        )

        yuanji = _yuanji_row()
        uncommitted = predict_design_mode_model(model, [yuanji], [UNCOMMITTED_STRUCTURE_MODE])
        forced_back = predict_design_mode_model(model, [yuanji], ["back_half"])
        self.assertLess(
            abs(float(uncommitted[0]) - 0.503),
            abs(float(forced_back[0]) - 0.503),
        )
        self.assertGreaterEqual(float(forced_back[0]), 0.5)

        v6_like = predict_design_mode_model(
            model,
            [yuanji],
            [UNCOMMITTED_STRUCTURE_MODE],
            ungated_fallback=np.asarray([0.503]),
        )
        self.assertAlmostEqual(float(v6_like[0]), 0.503)


if __name__ == "__main__":
    unittest.main()
