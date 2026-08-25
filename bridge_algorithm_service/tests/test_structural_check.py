from __future__ import annotations

import unittest

from bridge_algorithm_service.structural_check import compute_structural_check


class StructuralCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "span": 14.0,
            "width": 4.5,
            "n1": 8,
            "n2": 7,
        }
        self.result = {
            "rise_height": 2.8,
            "rise_span": 0.2,
            "s1_flat_lo": 240.0,
            "s1_flat_hi": 290.0,
            "s1_diag_lo": 270.0,
            "s1_diag_hi": 290.0,
            "s2_flat_lo": 190.0,
            "s2_flat_hi": 240.0,
            "s2_diag1_lo": 170.0,
            "s2_diag1_hi": 200.0,
            "s2_diag2_lo": 200.0,
            "s2_diag2_hi": 220.0,
            "s2_diag1_len": 3.2,
            "s2_diag2_len": 3.5,
            "s2_flat_len": 2.0,
            "geometry": {
                "five_miao_bullhead": {
                    "ratios": {"alpha_outer": 0.58, "beta_inner": 0.28},
                }
            },
        }

    def test_predicted_scheme_passes_screening(self) -> None:
        check = compute_structural_check(self.params, self.result)

        self.assertTrue(check["passes"])
        self.assertEqual(check["status"], "screening_pass")
        self.assertLess(check["max_dcr"], 1.0)
        self.assertEqual(len(check["members"]), 5)
        self.assertEqual(check["load_model"]["plane_systems"], 15)
        self.assertEqual(set(check["decay_life_proxy"]), {"protected", "sheltered", "exposed"})

    def test_undersized_members_fail_screening(self) -> None:
        undersized = dict(self.result)
        undersized.update({
            "s1_flat_lo": 40.0,
            "s1_flat_hi": 50.0,
            "s1_diag_lo": 40.0,
            "s1_diag_hi": 50.0,
            "s2_flat_lo": 30.0,
            "s2_flat_hi": 40.0,
            "s2_diag1_lo": 30.0,
            "s2_diag1_hi": 40.0,
            "s2_diag2_lo": 30.0,
            "s2_diag2_hi": 40.0,
        })

        check = compute_structural_check(self.params, undersized)

        self.assertFalse(check["passes"])
        self.assertEqual(check["status"], "screening_fail")
        self.assertGreater(check["max_dcr"], 1.0)


if __name__ == "__main__":
    unittest.main()
