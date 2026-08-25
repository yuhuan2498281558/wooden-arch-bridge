from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ml_pipeline.train.pilot import (
    choose_winner,
    group_macro_mae,
    load_training_rows,
    make_estimator,
    metric_record,
    parameter_grid,
    predict_artifact,
    tune_model,
)

from bridge_algorithm_service.tests import test_symmetric_targets as target_fixtures


class PilotTrainingTests(unittest.TestCase):
    def test_quantile_candidate_fits_and_predicts_finite_values(self) -> None:
        x = np.asarray([[10.0], [12.0], [14.0], [16.0], [18.0]])
        y = np.asarray([-0.10, -0.04, 0.0, 0.05, 0.11])

        self.assertTrue(parameter_grid("quantile"))
        estimator = make_estimator("quantile", {"alpha": 0.0})
        predicted = estimator.fit(x, y).predict(x)

        self.assertEqual(predicted.shape, y.shape)
        self.assertTrue(np.isfinite(predicted).all())

    def test_group_macro_mae_does_not_overweight_multi_span_bridge(self) -> None:
        observed = np.asarray([0.0, 0.0, 0.0, 1.0])
        predicted = np.asarray([1.0, 1.0, 1.0, 1.0])
        groups = np.asarray(["multi", "multi", "multi", "single"])

        self.assertAlmostEqual(group_macro_mae(observed, predicted, groups), 0.5)

    def test_metric_record_separates_bridge_and_split_group_macro_mae(self) -> None:
        observed = np.asarray([0.0, 0.0, 1.0])
        predicted = np.asarray([1.0, 1.0, 1.0])
        split_groups = np.asarray(["survey-family", "survey-family", "survey-family"])
        bridge_keys = np.asarray(["bridge-a", "bridge-a", "bridge-b"])

        record = metric_record(
            observed,
            predicted,
            split_groups,
            bridge_keys=bridge_keys,
            bootstrap_iterations=100,
            random_seed=7,
        )

        self.assertAlmostEqual(record["bridge_macro_mae"], 0.5)
        self.assertAlmostEqual(record["split_group_macro_mae"], 2.0 / 3.0)
        self.assertAlmostEqual(record["group_macro_mae"], record["bridge_macro_mae"])

    def test_tune_model_optimizes_bridge_macro_not_sample_mae(self) -> None:
        class FakeEstimator:
            def __init__(self, alpha: float) -> None:
                self.alpha = alpha

            def fit(self, _x: np.ndarray, _y: np.ndarray) -> "FakeEstimator":
                return self

            def predict(self, x: np.ndarray) -> np.ndarray:
                if self.alpha == 1.0:
                    return np.where(x[:, 0] == 0.0, 0.4, 0.0)
                return np.full(len(x), 0.15)

        x = np.asarray([[0.0], [0.0], [0.0], [0.0], [1.0], [2.0], [3.0]])
        residual = np.zeros(len(x))
        split_groups = np.asarray(["a", "a", "a", "a", "b", "c", "d"])
        bridge_keys = np.asarray(["a", "a", "a", "a", "b", "c", "d"])

        with patch("ml_pipeline.train.pilot.parameter_grid", return_value=[{"alpha": 1.0}, {"alpha": 2.0}]), patch(
            "ml_pipeline.train.pilot.make_estimator",
            side_effect=lambda _name, params: FakeEstimator(float(params["alpha"])),
        ):
            params, score = tune_model(
                "ridge",
                x,
                residual,
                split_groups,
                metric_groups=bridge_keys,
                inner_splits=4,
            )

        self.assertEqual(params, {"alpha": 1.0})
        self.assertAlmostEqual(score, 0.1)

    def test_choose_winner_uses_bridge_macro_mae(self) -> None:
        metrics = {
            "ridge": {"group_macro_mae": 0.04},
            "huber": {"group_macro_mae": 0.05},
            "fixed_rule": {"group_macro_mae": 0.09},
        }

        self.assertEqual(choose_winner(metrics), "ridge")

    def test_choose_winner_prefers_simpler_model_inside_tolerance(self) -> None:
        metrics = {
            "ridge": {"group_macro_mae": 0.0400},
            "huber": {"group_macro_mae": 0.0399},
            "train_median": {"group_macro_mae": 0.0600},
        }

        self.assertEqual(choose_winner(metrics), "ridge")

    def test_constant_artifact_prediction_is_projected_to_bounds(self) -> None:
        artifact = {
            "estimator": None,
            "constant": 0.6,
            "rule_baseline": 0.25,
            "bounds": [0.0, 0.5],
        }

        prediction = predict_artifact(artifact, np.asarray([[10.0], [20.0]]))

        np.testing.assert_allclose(prediction, [0.5, 0.5])

    def test_training_loader_excludes_zero_inner_flat_chord_with_reason(self) -> None:
        from json import dumps, loads
        from pathlib import Path
        from tempfile import TemporaryDirectory

        fixture = target_fixtures.SymmetricTargetPreparationTests()
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            paths = []
            for index in range(11):
                path = fixture._write_annotation(directory)
                payload = loads(path.read_text(encoding="utf-8"))
                payload["bridge"]["bridge_id"] = f"T-{index}"
                payload["bridge"]["source_group_id"] = f"group-{index}"
                output = directory / f"sample-{index}_five_miao_nodes.json"
                output.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
                paths.append(output)
            zero_path = paths[-1]
            payload = loads(zero_path.read_text(encoding="utf-8"))
            payload["annotation"]["points"]["Q2"] = {"x": 450, "y": -92}
            payload["annotation"]["points"]["Q3"] = {"x": 450, "y": -86}
            zero_path.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")

            all_rows, accepted = load_training_rows(paths)

        self.assertEqual(len(all_rows), 11)
        self.assertEqual(len(accepted), 10)
        excluded = next(row for row in all_rows if not row["training_eligible"])
        self.assertEqual(excluded["training_exclusion_reasons"], "zero_inner_flat_chord")

    def test_training_loader_soft_excludes_missing_span(self) -> None:
        from json import dumps, loads
        from pathlib import Path
        from tempfile import TemporaryDirectory

        fixture = target_fixtures.SymmetricTargetPreparationTests()
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            paths = []
            for index in range(11):
                path = fixture._write_annotation(directory)
                payload = loads(path.read_text(encoding="utf-8"))
                payload["bridge"]["bridge_id"] = f"T-{index}"
                payload["bridge"]["source_group_id"] = f"group-{index}"
                output = directory / f"sample-{index}_five_miao_nodes.json"
                output.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
                paths.append(output)
            missing_path = paths[-1]
            payload = loads(missing_path.read_text(encoding="utf-8"))
            del payload["span"]["clear_span_m"]
            del payload["bridge"]["span_m"]
            missing_path.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")

            all_rows, accepted = load_training_rows(paths)

        self.assertEqual(len(all_rows), 11)
        self.assertEqual(len(accepted), 10)
        excluded = next(row for row in all_rows if not row["training_eligible"])
        self.assertEqual(excluded["training_exclusion_reasons"], "missing_span_m")

    def test_training_loader_rejects_bridge_split_across_groups(self) -> None:
        from json import dumps, loads
        from pathlib import Path
        from tempfile import TemporaryDirectory

        fixture = target_fixtures.SymmetricTargetPreparationTests()
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            paths = []
            for index in range(11):
                path = fixture._write_annotation(directory)
                payload = loads(path.read_text(encoding="utf-8"))
                payload["bridge"]["bridge_id"] = f"T-{index}"
                payload["bridge"]["source_group_id"] = f"group-{index}"
                output = directory / f"sample-{index}_five_miao_nodes.json"
                output.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
                paths.append(output)
            # 同一桥（T-0）的第二跨被错误地分到另一个组
            conflict_path = directory / "sample-conflict_five_miao_nodes.json"
            payload = loads(paths[0].read_text(encoding="utf-8"))
            payload["bridge"]["bridge_id"] = "T-0"
            payload["bridge"]["source_group_id"] = "group-other"
            payload["span"] = {"index": 2, "count": 2, "clear_span_m": 18.6}
            payload["bridge"]["span_count"] = 2
            conflict_path.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
            paths.append(conflict_path)

            with self.assertRaisesRegex(ValueError, "must not be split across"):
                load_training_rows(paths)

    def test_training_loader_excludes_defective_rows_with_reason(self) -> None:
        from json import dumps, loads
        from pathlib import Path
        from tempfile import TemporaryDirectory

        fixture = target_fixtures.SymmetricTargetPreparationTests()
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            paths = []
            for index in range(11):
                path = fixture._write_annotation(directory)
                payload = loads(path.read_text(encoding="utf-8"))
                payload["bridge"]["bridge_id"] = f"T-{index}"
                payload["bridge"]["source_group_id"] = f"group-{index}"
                output = directory / f"sample-{index}_five_miao_nodes.json"
                output.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
                paths.append(output)
            defective_path = paths[-1]
            payload = loads(defective_path.read_text(encoding="utf-8"))
            payload["annotation"]["defects"] = ["node_unclear"]
            defective_path.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")

            all_rows, accepted = load_training_rows(paths)

        self.assertEqual(len(all_rows), 11)
        self.assertEqual(len(accepted), 10)
        excluded = next(row for row in all_rows if not row["training_eligible"])
        self.assertEqual(excluded["training_exclusion_reasons"], "defects")

    def test_training_loader_rejects_duplicate_sample_key(self) -> None:
        from json import dumps, loads
        from pathlib import Path
        from tempfile import TemporaryDirectory

        fixture = target_fixtures.SymmetricTargetPreparationTests()
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            paths = []
            for index in range(11):
                path = fixture._write_annotation(directory)
                payload = loads(path.read_text(encoding="utf-8"))
                payload["bridge"]["bridge_id"] = f"T-{index}"
                payload["bridge"]["source_group_id"] = f"group-{index}"
                output = directory / f"sample-{index}_five_miao_nodes.json"
                output.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
                paths.append(output)
            # 复制 sample-0 的桥/跨身份，产生重复 sample_key
            duplicate_path = directory / "sample-duplicate_five_miao_nodes.json"
            payload = loads(paths[0].read_text(encoding="utf-8"))
            duplicate_path.write_text(dumps(payload, ensure_ascii=False), encoding="utf-8")
            paths.append(duplicate_path)

            with self.assertRaisesRegex(ValueError, "duplicate sample_key"):
                load_training_rows(paths)


if __name__ == "__main__":
    unittest.main()
