from __future__ import annotations

import unittest

from bridge_algorithm_service.service_life import estimate_service_life


class ServiceLifeTests(unittest.TestCase):
    def test_prior_predictive_result_is_ordered_and_traceable(self) -> None:
        result = estimate_service_life("杉木")

        self.assertEqual(result["model_version"], "service-life-weibull-gamma-v1")
        self.assertEqual(result["update_mode"], "prior_predictive")
        self.assertFalse(result["posterior_updated"])
        self.assertLess(result["ci_90"][0], result["estimated_life"])
        self.assertLess(result["estimated_life"], result["ci_90"][1])
        self.assertGreater(result["remaining_life"], 0)

    def test_exposure_maintenance_and_treatment_change_lifetime_distribution(self) -> None:
        favourable = estimate_service_life("柏木", {
            "exposure_class": "protected",
            "maintenance_level": "good",
            "preservative_treatment": True,
        })
        adverse = estimate_service_life("柏木", {
            "exposure_class": "exposed",
            "maintenance_level": "poor",
            "preservative_treatment": False,
        })

        self.assertGreater(favourable["estimated_life"], adverse["estimated_life"])

    def test_current_age_is_used_as_right_censored_evidence(self) -> None:
        result = estimate_service_life("default", {
            "current_age": 45,
        })

        self.assertEqual(result["update_mode"], "right_censored_update")
        self.assertTrue(result["posterior_updated"])
        self.assertGreaterEqual(result["ci_90"][0], 45)
        self.assertAlmostEqual(
            result["remaining_life"],
            result["estimated_life"] - 45,
            places=1,
        )

    def test_uncalibrated_condition_label_does_not_change_lifetime(self) -> None:
        good = estimate_service_life("default", {"current_age": 20, "condition_state": "good"})
        poor = estimate_service_life("default", {"current_age": 20, "condition_state": "poor"})

        self.assertEqual(good["estimated_life"], poor["estimated_life"])
        self.assertIn("condition_state", good["factors_not_used"])

    def test_wood_common_name_does_not_create_an_unapproved_species_ranking(self) -> None:
        fir = estimate_service_life("杉木")
        pine = estimate_service_life("马尾松")

        self.assertEqual(fir["estimated_life"], pine["estimated_life"])

    def test_survival_probability_is_bounded(self) -> None:
        result = estimate_service_life("default", {"current_age": 20})

        self.assertGreaterEqual(result["survival_probability_next_10y"], 0)
        self.assertLessEqual(result["survival_probability_next_10y"], 1)


if __name__ == "__main__":
    unittest.main()
