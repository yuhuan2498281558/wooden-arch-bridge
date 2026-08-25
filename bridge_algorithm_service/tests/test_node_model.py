from __future__ import annotations

import unittest

from bridge_algorithm_service.node_model import (
    DesignModeAlphaModel,
    NodeModelOutOfDistribution,
    NodeRatioModelBundle,
)
from bridge_algorithm_service.back_half_high_tail import FIXED_RULE_ALPHA


def _artifact(target: str, constant: float, feature_names: list[str]) -> dict:
    return {
        "artifact_version": "five-miao-node-pilot-v5-test",
        "status": "pilot_not_for_production",
        "target": target,
        "model_name": "train_median",
        "feature_set": "test",
        "feature_names": feature_names,
        "rule_baseline": 2.0 / 3.0 if target == "alpha" else 0.25,
        "constant": constant,
        "estimator": None,
        "bounds": [0.0, 1.0 if target == "alpha" else 0.5],
        "training_feature_range": {
            "span_m": [10.0, 30.0],
            "three_miao_rise_span_ratio": [0.12, 0.25],
        },
        "oof_absolute_error_quantiles": {"q90": 0.05},
        "independent_holdout_metrics": {"bridge_macro_mae": 0.04},
    }


def _design_mode_artifact() -> dict:
    def expert(constant: float, bounds: list[float]) -> dict:
        return {
            "model_name": "median",
            "feature_set": "none",
            "feature_names": [],
            "params": {},
            "constant": constant,
            "estimator": None,
            "output_bounds": bounds,
            "training_feature_range": {},
            "applicability_domain": None,
        }

    return {
        "artifact_version": "five-miao-node-pilot-v9-designer-selected-mode",
        "status": "research_designer_mode_not_for_deployment",
        "target": "alpha",
        "deployment_mode_source_required": "designer_selected",
        "historical_mode_source": "derived_from_design_target_alpha_threshold",
        "integration_gate_passed": True,
        "experts": {
            "front_half": expert(0.4, [0.0, 0.5]),
            "back_half": expert(0.65, [0.5, 1.0]),
        },
        "development_metrics": {
            "by_mode": {
                "front_half": {"absolute_error_quantiles": {"q90": 0.08}},
                "back_half": {"absolute_error_quantiles": {"q90": 0.1}},
            }
        },
    }


class NodeModelTests(unittest.TestCase):
    def test_bundle_predicts_both_ratios_with_traceable_metadata(self) -> None:
        bundle = NodeRatioModelBundle(
            _artifact("alpha", 0.61, ["span_m"]),
            _artifact("beta", 0.22, ["span_m", "three_miao_rise_span_ratio"]),
        )

        decision = bundle.predict({"span_m": 18.0, "three_miao_rise_span_ratio": 0.18})

        self.assertEqual(decision.source, "pilot_model")
        self.assertAlmostEqual(decision.alpha, 0.61)
        self.assertAlmostEqual(decision.beta, 0.22)
        self.assertTrue(decision.metadata["within_training_range"])
        self.assertEqual(decision.metadata["model_status"], "pilot_not_for_production")

    def test_bundle_rejects_out_of_training_range_features(self) -> None:
        bundle = NodeRatioModelBundle(
            _artifact("alpha", 0.61, ["span_m"]),
            _artifact("beta", 0.22, ["span_m"]),
        )

        with self.assertRaisesRegex(NodeModelOutOfDistribution, "outside"):
            bundle.predict({"span_m": 50.0})

    def test_bundle_marks_sparse_local_region_without_discarding_prediction(self) -> None:
        alpha = _artifact("alpha", 0.61, ["span_m"])
        beta = _artifact("beta", 0.22, ["span_m"])
        domain = {
            "method": "standardized-mean-k-nearest-distance-v1",
            "center": [20.0],
            "scale": [5.0],
            "normalized_training_points": [[-2.0], [-1.8], [1.8], [2.0]],
            "nearest_neighbor_k": 2,
            "sparse_region_threshold": 0.5,
        }
        alpha["applicability_domain"] = domain
        beta["applicability_domain"] = domain
        bundle = NodeRatioModelBundle(alpha, beta)

        decision = bundle.predict({"span_m": 20.0})

        self.assertEqual(decision.source, "pilot_model")
        self.assertFalse(decision.metadata["within_local_domain"])
        self.assertIn("sparse_training_region:alpha", decision.metadata["model_warnings"])
        interval = decision.metadata["target_models"]["alpha"]["prediction_interval_90"]
        self.assertAlmostEqual(interval[0], 0.56)
        self.assertAlmostEqual(interval[1], 0.66)

    def test_design_mode_model_requires_explicit_supported_mode(self) -> None:
        model = DesignModeAlphaModel(_design_mode_artifact())

        front, front_metadata = model.predict({}, "front_half")
        back, back_metadata = model.predict({}, "back_half")

        self.assertAlmostEqual(front, 0.4)
        self.assertAlmostEqual(back, 0.65)
        self.assertLess(front, 0.5)
        self.assertGreaterEqual(back, 0.5)
        self.assertEqual(front_metadata["prediction_interval_90"], [0.32, 0.48000000000000004])
        self.assertEqual(back_metadata["prediction_interval_90"], [0.55, 0.75])
        self.assertFalse(back_metadata["high_tail_mix_applied"])
        self.assertFalse(front_metadata["low_tail_mix_applied"])
        with self.assertRaisesRegex(Exception, "unsupported outer-node design mode"):
            model.predict({}, "automatic")

    def test_design_mode_back_half_mixes_yonggui_like_shrinkage_to_two_thirds(self) -> None:
        artifact = _design_mode_artifact()
        artifact["experts"]["back_half"]["constant"] = 0.597
        artifact["back_half_high_tail"] = {
            "method": "rule_floor_when_expert_below_median_by_shrink_band",
            "rule_alpha": FIXED_RULE_ALPHA,
            "train_median": 0.622,
            "shrink_band": 0.02,
            "high_tail_alpha": 0.70,
            "high_tail_train_rows": 7,
        }
        model = DesignModeAlphaModel(artifact)

        back, metadata = model.predict({}, "back_half")
        front, _ = model.predict({}, "front_half")

        self.assertGreaterEqual(back, FIXED_RULE_ALPHA - 1e-12)
        self.assertTrue(metadata["high_tail_mix_applied"])
        self.assertAlmostEqual(front, 0.4)
        self.assertLess(front, 0.5)

    def test_design_mode_front_half_does_not_mix_lanxia_like_low_tail(self) -> None:
        artifact = _design_mode_artifact()
        artifact["experts"]["front_half"]["constant"] = 0.427
        artifact["front_half_low_tail"] = {
            "method": "none_isolated_holdout_outlier",
            "enabled": False,
            "train_median": 0.449,
            "low_tail_alpha": 0.30,
            "low_tail_train_rows": 1,
        }
        model = DesignModeAlphaModel(artifact)

        front, metadata = model.predict({}, "front_half")
        back, _ = model.predict({}, "back_half")

        self.assertAlmostEqual(front, 0.427)
        self.assertFalse(metadata["low_tail_mix_applied"])
        self.assertLess(front, 0.5)
        self.assertGreater(front, 0.35)
        self.assertAlmostEqual(back, 0.65)

    def test_design_mode_uncommitted_uses_band_center_not_back_expert(self) -> None:
        artifact = _design_mode_artifact()
        artifact["uncommitted_alpha"] = {
            "method": "boundary_band_center_else_ungated_expert_mix",
            "band_center": 0.5,
            "band_alpha": 0.5,
            "boundary_band": 0.03,
            "band_rows": 22,
            "off_band_uncommitted_rows": 3,
        }
        model = DesignModeAlphaModel(artifact)

        uncommitted, metadata = model.predict({}, "uncommitted")
        back, _ = model.predict({}, "back_half")

        self.assertAlmostEqual(uncommitted, 0.5)
        self.assertEqual(metadata["uncommitted_predictor"], "boundary_band_center")
        self.assertGreater(abs(back - 0.5), 0.1)

    def test_design_mode_model_rejects_artifact_that_failed_gate(self) -> None:
        artifact = _design_mode_artifact()
        artifact["integration_gate_passed"] = False

        with self.assertRaisesRegex(Exception, "did not pass"):
            DesignModeAlphaModel(artifact)


if __name__ == "__main__":
    unittest.main()
