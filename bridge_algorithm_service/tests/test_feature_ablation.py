from __future__ import annotations

import unittest

from ml_pipeline.train.feature_ablation import choose_feature_model, split_development_holdout


class FeatureAblationTests(unittest.TestCase):
    def test_prefers_smaller_feature_set_inside_selection_tolerance(self) -> None:
        metrics = {
            "span_only": {"alpha": {"ridge": {"group_macro_mae": 0.0500}}},
            "span_rise_ratio": {"alpha": {"ridge": {"group_macro_mae": 0.0490}}},
        }

        self.assertEqual(choose_feature_model(metrics, "alpha"), ("span_only", "ridge"))

    def test_selects_richer_feature_set_for_material_improvement(self) -> None:
        metrics = {
            "span_only": {"beta": {"ridge": {"group_macro_mae": 0.0500}}},
            "span_rise_ratio": {"beta": {"ridge": {"group_macro_mae": 0.0470}}},
        }

        self.assertEqual(choose_feature_model(metrics, "beta"), ("span_rise_ratio", "ridge"))

    def test_development_holdout_keeps_split_groups_disjoint(self) -> None:
        rows = [
            {"sample_key": f"sample-{index}", "split_group_key": f"group-{index // 2}"}
            for index in range(20)
        ]

        development, holdout = split_development_holdout(
            rows,
            holdout_fraction=0.2,
            random_seed=17,
        )

        development_groups = {row["split_group_key"] for row in development}
        holdout_groups = {row["split_group_key"] for row in holdout}
        self.assertFalse(development_groups & holdout_groups)
        self.assertEqual(len(development) + len(holdout), len(rows))


if __name__ == "__main__":
    unittest.main()
