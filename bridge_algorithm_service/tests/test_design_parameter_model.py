from __future__ import annotations

import unittest

from bridge_algorithm_service.design_parameter_model import (
    MentorDesignParameterModel,
    design_parameter_model_status,
)


class MentorDesignParameterModelTests(unittest.TestCase):
    def test_packaged_model_is_available_and_hash_verified(self) -> None:
        status = design_parameter_model_status()

        self.assertTrue(status["enabled"])
        self.assertTrue(status["available"])
        self.assertEqual(status["model_version"], "mentor-two-stage-2026-04-08")
        self.assertEqual(status["stage1"], "SSA-XGBoost")
        self.assertEqual(status["stage2"], "CF-BPNN")

    def test_prediction_replays_legacy_two_stage_snapshot(self) -> None:
        result = MentorDesignParameterModel().predict(
            length=32,
            width=4.5,
            span=15,
            n1=9,
            n2=8,
        )

        expected = {
            "rise_span": 0.1703,
            "rise_height": 2.554,
            "s1_flat_lo": 251.3,
            "s1_flat_hi": 290.0,
            "s1_diag_lo": 273.3,
            "s1_diag_hi": 290.0,
            "s2_flat_lo": 190.0,
            "s2_flat_hi": 235.7,
            "s2_diag1_lo": 170.0,
            "s2_diag1_hi": 191.5,
            "s2_diag2_lo": 200.0,
            "s2_diag2_hi": 202.1,
        }
        self.assertEqual({key: result[key] for key in expected}, expected)
        self.assertEqual(result["validation"]["method"], "SSA-XGBoost + CF-BPNN")
        self.assertEqual(result["validation"]["rise_input_mode"], "legacy_span_ratio_seed")

    def test_provided_rise_uses_the_legacy_feature_contract(self) -> None:
        result = MentorDesignParameterModel().predict(
            length=32,
            width=4.5,
            span=15,
            n1=9,
            n2=8,
            rise=2.7,
        )

        self.assertEqual(result["rise_span"], 0.171)
        self.assertEqual(result["rise_height"], 2.565)
        self.assertEqual(result["validation"]["rise_input_mode"], "provided")


if __name__ == "__main__":
    unittest.main()
