from __future__ import annotations

import unittest

from bridge_algorithm_service.safety_integration import (
    _structural_score_from_dcr,
    integrate_safety_analysis,
    select_decay_profile,
)


class SafetyIntegrationTests(unittest.TestCase):
    def test_structural_score_uses_load_factor_not_linear_dcr_penalty(self) -> None:
        # DCR 0.27 means capacity is 3.7x the assumed design load and must
        # score high, not around 67.
        self.assertGreater(_structural_score_from_dcr(0.27), 90)
        self.assertAlmostEqual(_structural_score_from_dcr(1.0), 60.0, places=1)
        self.assertLess(_structural_score_from_dcr(1.3), 60.0)

    def test_integration_uses_governing_life_and_explicit_weights(self) -> None:
        service_life = {
            "estimated_life": 40.0,
            "remaining_life": 40.0,
            "ci_90": [28.0, 48.0],
        }
        structural_check = {
            "max_dcr": 0.27,
            "selected_decay_life": {
                "critical_member": "五节苗内斜弦",
                "years_until_dcr_1": 68.0,
            },
        }
        decay_profile = select_decay_profile({
            "exposure_class": "sheltered",
            "maintenance_level": "regular",
            "preservative_treatment": True,
        })

        integrated = integrate_safety_analysis(
            service_life=service_life,
            structural_check=structural_check,
            decay_profile=decay_profile,
        )

        self.assertEqual(integrated["governing_life_years"], 40.0)
        self.assertEqual(integrated["governing_life_source"], "bayesian_prior")
        self.assertGreater(integrated["overall_score"]["score"], 85)
        self.assertEqual(len(integrated["chain"]), 4)


if __name__ == "__main__":
    unittest.main()
