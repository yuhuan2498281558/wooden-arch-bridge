from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml_pipeline.train.model_improvement import (
    choose_screening_candidate,
    load_frozen_partition,
    passes_replacement_gate,
)


def _metric(mae: float, q90: float) -> dict:
    return {
        "bridge_macro_mae": mae,
        "group_macro_mae": mae,
        "absolute_error_quantiles": {"q90": q90},
    }


class ModelImprovementTests(unittest.TestCase):
    def test_replacement_gate_requires_material_mae_gain(self) -> None:
        passed, reasons = passes_replacement_gate(
            _metric(0.079, 0.12),
            _metric(0.080, 0.12),
        )

        self.assertFalse(passed)
        self.assertIn("bridge_macro_mae_not_materially_better", reasons)

    def test_replacement_gate_rejects_worse_tail(self) -> None:
        passed, reasons = passes_replacement_gate(
            _metric(0.070, 0.14),
            _metric(0.080, 0.12),
        )

        self.assertFalse(passed)
        self.assertIn("q90_tail_error_worse", reasons)

    def test_replacement_gate_accepts_robust_improvement(self) -> None:
        passed, reasons = passes_replacement_gate(
            _metric(0.070, 0.115),
            _metric(0.080, 0.12),
        )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_screening_prefers_simpler_candidate_inside_tolerance(self) -> None:
        model_names = (
            "ridge",
            "huber",
            "elastic_net",
            "spline_ridge",
            "svr",
            "gpr",
            "extra_trees",
        )
        metrics = {}
        for feature_set in ("span_only", "span_rise_ratio"):
            metrics[feature_set] = {"alpha": {name: _metric(0.2, 0.2) for name in model_names}}
        metrics["span_only"]["alpha"]["ridge"] = _metric(0.081, 0.12)
        metrics["span_rise_ratio"]["alpha"]["svr"] = _metric(0.080, 0.11)

        self.assertEqual(choose_screening_candidate(metrics, "alpha"), ("span_only", "ridge"))

    def test_frozen_partition_is_exact_and_group_disjoint(self) -> None:
        rows = [
            {"sample_key": "a", "split_group_key": "g1"},
            {"sample_key": "b", "split_group_key": "g2"},
            {"sample_key": "c", "split_group_key": "g3"},
        ]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "partition.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_key", "partition"])
                writer.writeheader()
                writer.writerows([
                    {"sample_key": "a", "partition": "development"},
                    {"sample_key": "b", "partition": "development"},
                    {"sample_key": "c", "partition": "independent_holdout"},
                ])

            development, holdout = load_frozen_partition(rows, path)

        self.assertEqual([row["sample_key"] for row in development], ["a", "b"])
        self.assertEqual([row["sample_key"] for row in holdout], ["c"])


if __name__ == "__main__":
    unittest.main()
