from __future__ import annotations

import unittest

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from bridge_algorithm_service.back_half_high_tail import (
    BACK_HALF_HIGH_TAIL_ALPHA,
    BACK_HALF_SHRINK_BAND,
    FIXED_RULE_ALPHA,
    mix_back_half_high_tail_alpha,
)
from bridge_algorithm_service.front_half_low_tail import (
    FRONT_HALF_LOW_TAIL_MIX_ENABLED,
    FRONT_HALF_TRAIN_MEDIAN,
    LANXIA_OBSERVED_ALPHA,
    LANXIA_OBSERVED_LEFT,
    LANXIA_OBSERVED_RIGHT,
    LANXIA_SPAN_M,
    LANXIA_V9_FRONT_EXPERT,
    copied_high_tail_trigger_would_fire,
    mix_front_half_low_tail_alpha,
)
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


def _yonggui_row() -> dict:
    # 咏归 independent_holdout: span backfilled 21.7 m, L/R 0.755/0.741, mean 0.748.
    return {
        "span_m": 21.7,
        "three_miao_rise_span_ratio": 0.19,
        "design_target_alpha": 0.748,
        "observed_alpha_left": 0.755,
        "observed_alpha_right": 0.741,
        "sample_key": "yonggui",
        "split_group_key": "g-yonggui",
        "bridge_key": "b-yonggui",
        "bridge_name": "咏归桥",
    }


def _median_back_half_row() -> dict:
    return {
        "span_m": 16.0,
        "three_miao_rise_span_ratio": 0.19,
        "design_target_alpha": 0.629,
        "observed_alpha_left": 0.630,
        "observed_alpha_right": 0.628,
        "sample_key": "median-back",
        "split_group_key": "g-median-back",
        "bridge_key": "b-median-back",
        "bridge_name": "median-back-half",
    }


def _lanxia_like_row() -> dict:
    # 岚下 2026-08-25 re-annotation: span 15.8 m, L/R 0.277/0.187, mean 0.232.
    return {
        "span_m": LANXIA_SPAN_M,
        "three_miao_rise_span_ratio": 0.19,
        "design_target_alpha": LANXIA_OBSERVED_ALPHA,
        "observed_alpha_left": LANXIA_OBSERVED_LEFT,
        "observed_alpha_right": LANXIA_OBSERVED_RIGHT,
        "sample_key": "lanxia",
        "split_group_key": "g-lanxia",
        "bridge_key": "b-lanxia",
        "bridge_name": "岚下桥",
    }


def _median_front_half_row() -> dict:
    return {
        "span_m": 16.0,
        "three_miao_rise_span_ratio": 0.19,
        "design_target_alpha": 0.449,
        "observed_alpha_left": 0.452,
        "observed_alpha_right": 0.446,
        "sample_key": "median-front",
        "split_group_key": "g-median-front",
        "bridge_key": "b-median-front",
        "bridge_name": "median-front-half",
    }


class _ShrinkBackHalfExpert:
    """Span-only map matching v9 咏归 shrinkage: 21.7 m → ~0.597, 16 m → ~0.630."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        span = np.asarray(X, dtype=float)[:, 0]
        return 0.722 - 0.00576 * span


def _shrinking_back_half_model() -> dict:
    model = fit_design_mode_model(_rows(), _median_specs())
    model["experts"]["back_half"] = {
        "model_name": "ridge",
        "feature_set": "span_only",
        "feature_names": ["span_m"],
        "params": {"alpha": 1e-8},
        "constant": None,
        "estimator": _ShrinkBackHalfExpert(),
    }
    # Verified committed back-half median on the 103-sample set.
    model["back_half_high_tail"]["train_median"] = 0.622
    model["back_half_high_tail"]["shrink_band"] = BACK_HALF_SHRINK_BAND
    model["back_half_high_tail"]["rule_alpha"] = FIXED_RULE_ALPHA
    return model


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

    def test_yonggui_like_back_half_is_lifted_off_bulk_shrinkage(self) -> None:
        yonggui = _yonggui_row()
        median_row = _median_back_half_row()
        lanxia = _lanxia_like_row()
        yuanji = _yuanji_row()
        self.assertEqual(int(_classes([yonggui])[0]), 1)
        self.assertEqual(int(_classes([median_row])[0]), 1)
        self.assertEqual(int(_classes([lanxia])[0]), 0)
        self.assertEqual(int(_classes([yuanji])[0]), -1)

        model = _shrinking_back_half_model()
        raw_yonggui = float(_ShrinkBackHalfExpert().predict(np.asarray([[21.7]]))[0])
        raw_median = float(_ShrinkBackHalfExpert().predict(np.asarray([[16.0]]))[0])
        self.assertAlmostEqual(raw_yonggui, 0.597, places=3)
        self.assertLess(raw_yonggui, 0.62)

        predicted = predict_design_mode_model(
            model,
            [yonggui, median_row, lanxia, yuanji],
            ["back_half", "back_half", "front_half", UNCOMMITTED_STRUCTURE_MODE],
        )
        yonggui_hat, median_hat, lanxia_hat, yuanji_hat = (float(value) for value in predicted)

        self.assertGreater(yonggui_hat, raw_yonggui)
        self.assertGreaterEqual(yonggui_hat, FIXED_RULE_ALPHA - 1e-12)
        self.assertLess(
            abs(yonggui_hat - 0.748),
            abs(raw_yonggui - 0.748),
        )
        self.assertLessEqual(abs(yonggui_hat - 0.748), abs(FIXED_RULE_ALPHA - 0.748) + 1e-12)

        self.assertAlmostEqual(median_hat, raw_median, places=5)
        self.assertLess(median_hat, 0.70)
        self.assertNotAlmostEqual(median_hat, 0.75, places=2)

        self.assertLess(lanxia_hat, 0.5)
        self.assertLess(yuanji_hat, 0.55)
        self.assertNotAlmostEqual(yuanji_hat, FIXED_RULE_ALPHA, places=2)
        self.assertLess(abs(yuanji_hat - 0.503), abs(yonggui_hat - 0.503))

    def test_high_tail_mix_only_fires_on_material_below_median_shrinkage(self) -> None:
        config = {
            "train_median": 0.622,
            "shrink_band": BACK_HALF_SHRINK_BAND,
            "rule_alpha": FIXED_RULE_ALPHA,
        }
        yonggui, yonggui_applied = mix_back_half_high_tail_alpha(0.597, config)
        median, median_applied = mix_back_half_high_tail_alpha(0.630, config)
        already_high, high_applied = mix_back_half_high_tail_alpha(0.70, config)
        missing, missing_applied = mix_back_half_high_tail_alpha(0.597, None)

        self.assertTrue(yonggui_applied)
        self.assertAlmostEqual(yonggui, FIXED_RULE_ALPHA)
        self.assertFalse(median_applied)
        self.assertAlmostEqual(median, 0.630)
        self.assertFalse(high_applied)
        self.assertAlmostEqual(already_high, 0.70)
        self.assertFalse(missing_applied)
        self.assertAlmostEqual(missing, 0.597)
        self.assertEqual(BACK_HALF_HIGH_TAIL_ALPHA, 0.70)
        self.assertEqual(BACK_HALF_SHRINK_BAND, 0.02)

    def test_lanxia_like_stays_front_half_without_low_tail_mix(self) -> None:
        lanxia = _lanxia_like_row()
        median_front = _median_front_half_row()
        yonggui = _yonggui_row()
        yuanji = _yuanji_row()
        self.assertEqual(int(_classes([lanxia])[0]), 0)
        self.assertEqual(int(_classes([median_front])[0]), 0)
        self.assertEqual(int(_classes([yonggui])[0]), 1)
        self.assertEqual(int(_classes([yuanji])[0]), -1)
        self.assertFalse(FRONT_HALF_LOW_TAIL_MIX_ENABLED)

        naive_would_fire = copied_high_tail_trigger_would_fire(LANXIA_V9_FRONT_EXPERT)
        self.assertTrue(naive_would_fire)
        self.assertLess(FRONT_HALF_TRAIN_MEDIAN - LANXIA_V9_FRONT_EXPERT, 0.03)
        unchanged, applied = mix_front_half_low_tail_alpha(LANXIA_V9_FRONT_EXPERT)
        self.assertFalse(applied)
        self.assertAlmostEqual(unchanged, LANXIA_V9_FRONT_EXPERT)

        model = fit_design_mode_model(_rows(), _median_specs())
        model["experts"]["front_half"] = {
            "model_name": "median",
            "feature_set": "none",
            "feature_names": [],
            "params": {},
            "constant": LANXIA_V9_FRONT_EXPERT,
            "estimator": None,
        }
        self.assertFalse(model["front_half_low_tail"]["enabled"])

        predicted = predict_design_mode_model(
            model,
            [lanxia, median_front, yonggui, yuanji],
            ["front_half", "front_half", "back_half", UNCOMMITTED_STRUCTURE_MODE],
        )
        lanxia_hat, median_hat, yonggui_hat, yuanji_hat = (float(value) for value in predicted)

        self.assertAlmostEqual(lanxia_hat, LANXIA_V9_FRONT_EXPERT)
        self.assertAlmostEqual(median_hat, LANXIA_V9_FRONT_EXPERT)
        self.assertGreater(lanxia_hat, 0.35)
        self.assertLess(lanxia_hat, 0.5)
        self.assertNotAlmostEqual(median_hat, 0.25, places=2)
        self.assertGreaterEqual(yonggui_hat, 0.5)
        self.assertEqual(int(_classes([yuanji])[0]), -1)
        self.assertLess(yuanji_hat, 0.55)


if __name__ == "__main__":
    unittest.main()
