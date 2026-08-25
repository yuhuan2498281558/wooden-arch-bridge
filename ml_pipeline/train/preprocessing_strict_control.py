# -*- coding: utf-8 -*-
"""Strict paired control for raw node labels versus the design preprocessing.

The comparison freezes samples, group splits, features, estimator family and
hyperparameters.  The only changed input is the label construction:

* raw-label baseline: fit the left/right observations independently and average
  their predictions to obtain a symmetric design output;
* full preprocessing: fit the parallel-angle-corrected, left/right-mean design
  target directly.

Scope filtering and span metadata are intentionally held fixed.  Turning those
off would change the sample population and would no longer be a paired control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.pilot import load_training_rows


RIDGE_ALPHA = 1e-8
RANDOM_SEED = 20260820
FEATURES = {
    "alpha": ("span_m",),
    "beta": ("span_m", "three_miao_rise_span_ratio"),
}
BOUNDS = {"alpha": (0.0, 1.0), "beta": (0.0, 0.5)}
DEFAULT_DATA_DIR = Path(r"C:\Users\24982\Desktop\数据集")
DEFAULT_PARTITION = Path(
    r"C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v5_independent_holdout\data_partition.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\24982\Desktop\机器学习结果\preprocessing_strict_control_v1"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _feature_array(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in FEATURES[target]] for row in rows],
        dtype=float,
    )


def _targets(rows: list[dict[str, Any]], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target == "alpha":
        left = np.asarray([float(row["legacy_projected_alpha_left"]) for row in rows])
        right = np.asarray([float(row["legacy_projected_alpha_right"]) for row in rows])
    else:
        left = np.asarray([float(row["observed_beta_left"]) for row in rows])
        right = np.asarray([float(row["observed_beta_right"]) for row in rows])
    processed = np.asarray([float(row[f"design_target_{target}"]) for row in rows])
    return left, right, processed


def _fit_predict(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    target: str,
) -> dict[str, np.ndarray]:
    x_train = _feature_array(train_rows, target)
    x_test = _feature_array(test_rows, target)
    raw_left, raw_right, processed = _targets(train_rows, target)

    processed_model = _model().fit(x_train, processed)
    left_model = _model().fit(x_train, raw_left)
    right_model = _model().fit(x_train, raw_right)

    processed_pred = processed_model.predict(x_test)
    raw_left_pred = left_model.predict(x_test)
    raw_right_pred = right_model.predict(x_test)
    return {
        "processed": processed_pred,
        "raw_left": raw_left_pred,
        "raw_right": raw_right_pred,
        "raw_averaged": (raw_left_pred + raw_right_pred) / 2.0,
    }


def _logo_predictions(rows: list[dict[str, Any]], target: str) -> dict[str, np.ndarray]:
    groups = np.asarray([str(row["split_group_key"]) for row in rows], dtype=object)
    out = {
        name: np.full(len(rows), np.nan, dtype=float)
        for name in ("processed", "raw_left", "raw_right", "raw_averaged")
    }
    for group in np.unique(groups):
        train_indexes = np.flatnonzero(groups != group)
        test_indexes = np.flatnonzero(groups == group)
        fitted = _fit_predict(
            [rows[index] for index in train_indexes],
            [rows[index] for index in test_indexes],
            target,
        )
        for name, values in fitted.items():
            out[name][test_indexes] = values
    return out


def _bridge_macro(values: np.ndarray, bridge_keys: np.ndarray) -> float:
    return float(np.mean([
        float(np.mean(values[bridge_keys == key]))
        for key in np.unique(bridge_keys)
    ]))


def _physical_error(target: str, errors: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    scales = []
    for row in rows:
        span = float(row["span_m"])
        if target == "alpha":
            rise = span * float(row["three_miao_rise_span_ratio"])
            scales.append(math.hypot(span / 3.0, rise))
        else:
            scales.append(span / 3.0)
    return errors * np.asarray(scales, dtype=float)


def _metrics(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = np.abs(y_true - y_pred)
    bridge_keys = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
    physical = _physical_error(target, errors, rows)
    lower, upper = BOUNDS[target]
    return {
        "n": len(rows),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bridge_macro_mae": _bridge_macro(errors, bridge_keys),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "q50_absolute_error": float(np.quantile(errors, 0.5)),
        "q90_absolute_error": float(np.quantile(errors, 0.9)),
        "max_absolute_error": float(np.max(errors)),
        "mean_node_error_m": float(np.mean(physical)),
        "raw_violation_count": int(np.sum((y_pred < lower) | (y_pred > upper))),
    }


def _paired_group_bootstrap(
    raw_errors: np.ndarray,
    processed_errors: np.ndarray,
    rows: list[dict[str, Any]],
    iterations: int = 5000,
) -> dict[str, float]:
    bridge_keys = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
    unique = np.unique(bridge_keys)
    raw_by_bridge = np.asarray([
        np.mean(raw_errors[bridge_keys == key]) for key in unique
    ])
    processed_by_bridge = np.asarray([
        np.mean(processed_errors[bridge_keys == key]) for key in unique
    ])
    paired = raw_by_bridge - processed_by_bridge
    rng = np.random.default_rng(RANDOM_SEED)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(paired), size=len(paired))
        draws[index] = float(np.mean(paired[sampled]))
    return {
        "raw_minus_processed_bridge_macro_mae": float(np.mean(paired)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_processed_better": float(np.mean(draws > 0.0)),
    }


def _evaluate(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    target: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_left, raw_right, y_true = _targets(rows, target)
    processed_pred = predictions["processed"]
    raw_pred = predictions["raw_averaged"]
    processed_errors = np.abs(y_true - processed_pred)
    raw_errors = np.abs(y_true - raw_pred)

    processed_metrics = _metrics(target, y_true, processed_pred, rows)
    raw_metrics = _metrics(target, y_true, raw_pred, rows)
    raw_bridge = raw_metrics["bridge_macro_mae"]
    improvement = (
        (raw_bridge - processed_metrics["bridge_macro_mae"]) / raw_bridge
        if raw_bridge > 0 else 0.0
    )
    side_error_raw = np.concatenate([
        np.abs(raw_left - predictions["raw_left"]),
        np.abs(raw_right - predictions["raw_right"]),
    ])
    side_error_processed = np.concatenate([
        np.abs(raw_left - processed_pred),
        np.abs(raw_right - processed_pred),
    ])
    predicted_asymmetry = np.abs(predictions["raw_left"] - predictions["raw_right"])
    record = {
        "processed": processed_metrics,
        "raw_labels": raw_metrics,
        "processed_bridge_macro_mae_improvement_rate": float(improvement),
        "paired_bridge_bootstrap": _paired_group_bootstrap(
            raw_errors, processed_errors, rows
        ),
        "raw_observation_side_mae": {
            "processed_symmetric_output": float(np.mean(side_error_processed)),
            "raw_side_models": float(np.mean(side_error_raw)),
        },
        "raw_side_prediction_asymmetry": {
            "mean": float(np.mean(predicted_asymmetry)),
            "q90": float(np.quantile(predicted_asymmetry, 0.9)),
            "max": float(np.max(predicted_asymmetry)),
            "rate_above_0_03": float(np.mean(predicted_asymmetry > 0.03)),
            "processed_output_asymmetry": 0.0,
        },
        "label_shift": {
            "mean_abs_raw_mean_minus_processed": float(
                np.mean(np.abs((raw_left + raw_right) / 2.0 - y_true))
            ),
            "max_abs_raw_mean_minus_processed": float(
                np.max(np.abs((raw_left + raw_right) / 2.0 - y_true))
            ),
        },
    }
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "sample_key": row["sample_key"],
            "bridge_key": row["bridge_key"],
            "split_group_key": row["split_group_key"],
            "target": target,
            "y_true_processed": float(y_true[index]),
            "raw_left_observed": float(raw_left[index]),
            "raw_right_observed": float(raw_right[index]),
            "processed_prediction": float(processed_pred[index]),
            "raw_left_prediction": float(predictions["raw_left"][index]),
            "raw_right_prediction": float(predictions["raw_right"][index]),
            "raw_averaged_prediction": float(raw_pred[index]),
            "processed_absolute_error": float(processed_errors[index]),
            "raw_absolute_error": float(raw_errors[index]),
            "raw_prediction_asymmetry": float(predicted_asymmetry[index]),
        })
    return record, prediction_rows


def _partition_rows(
    rows: list[dict[str, Any]], partition_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with partition_path.open(encoding="utf-8-sig") as handle:
        partition = {
            item["sample_key"]: item["partition"] for item in csv.DictReader(handle)
        }
    missing = [row["sample_key"] for row in rows if row["sample_key"] not in partition]
    if missing:
        raise ValueError(f"partition is missing {len(missing)} accepted rows")
    development = [row for row in rows if partition[row["sample_key"]] == "development"]
    holdout = [row for row in rows if partition[row["sample_key"]] == "independent_holdout"]
    if not development or not holdout:
        raise ValueError("partition must contain development and independent_holdout rows")
    dev_groups = {str(row["split_group_key"]) for row in development}
    holdout_groups = {str(row["split_group_key"]) for row in holdout}
    if dev_groups & holdout_groups:
        raise ValueError("split_group_key leakage between development and holdout")
    return development, holdout


def _write_predictions(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# 原始标签与完整预处理：严格配对对照实验",
        "",
        "- 冻结项：样本、split_group_key、特征、Ridge(alpha=1e-8) 与评估目标。",
        "- 原始标签基线：左右原始观测分别拟合，再取预测均值形成设计输出；不做平行角度修正。",
        "- 完整预处理：平行角度修正后取左右均值，直接拟合对称设计目标。",
        "- 范围过滤和净跨元数据保持一致；否则样本变化会破坏配对比较。",
        "",
    ]
    for split_name, title in (("development_logo", "开发集 LOOGO"), ("independent_holdout", "独立内部留出")):
        lines.extend([
            f"## {title}",
            "",
            "| 目标 | 原始标签桥级宏 MAE | 完整预处理桥级宏 MAE | 相对变化 | 配对差值 95% CI |",
            "|---|---:|---:|---:|---:|",
        ])
        for target in ("alpha", "beta"):
            item = result[split_name][target]
            rate = item["processed_bridge_macro_mae_improvement_rate"]
            boot = item["paired_bridge_bootstrap"]
            lines.append(
                f"| {target} | {item['raw_labels']['bridge_macro_mae']:.5f} | "
                f"{item['processed']['bridge_macro_mae']:.5f} | {rate:+.2%} | "
                f"[{boot['ci95_low']:.5f}, {boot['ci95_high']:.5f}] |"
            )
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- beta 的设计目标本来就是左右观测均值；在线性 Ridge 下，分别拟合后再平均与直接拟合均值理论上等价，因此不能期待仅靠对称化获得 MAE 增益。",
        "- alpha 的差异主要来自平行角度修正。该修正首先保证目标对应设计构造语义，精度变化是次级证据。",
        "- 共同评估目标仍是预处理生成的对称设计代理，并非独立专家真值；该对照验证标签构造一致性，不能单独证明更接近原始设计意图。",
        "- 这是同一网站数据源上的内部验证，不是独立外部验证。论文不得写成无条件泛化结论。",
    ])
    return "\n".join(lines) + "\n"


def run_experiment(
    data_dir: Path,
    partition_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    paths = collect_annotation_paths([], [data_dir], "*_five_miao_nodes.json")
    _, accepted = load_training_rows(paths)
    development, holdout = _partition_rows(accepted, partition_path)

    result: dict[str, Any] = {
        "protocol": {
            "comparison": "paired raw labels versus full preprocessing",
            "model": "StandardScaler + Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "features": FEATURES,
            "development_validation": "Leave-One-Group-Out by split_group_key",
            "holdout": "frozen v5 independent internal holdout",
            "fixed_controls": [
                "same accepted sample population",
                "same split_group_key folds",
                "same features",
                "same estimator and hyperparameter",
                "same processed design target for evaluation",
            ],
        },
        "data_dir": str(data_dir),
        "partition_path": str(partition_path),
        "accepted_rows": len(accepted),
        "accepted_groups": len({str(row["split_group_key"]) for row in accepted}),
        "development_rows": len(development),
        "development_groups": len({str(row["split_group_key"]) for row in development}),
        "holdout_rows": len(holdout),
        "holdout_groups": len({str(row["split_group_key"]) for row in holdout}),
        "development_logo": {},
        "independent_holdout": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for target in ("alpha", "beta"):
        dev_predictions = _logo_predictions(development, target)
        dev_record, dev_rows = _evaluate(development, dev_predictions, target)
        result["development_logo"][target] = dev_record
        prediction_rows.extend({"evaluation": "development_logo", **row} for row in dev_rows)

        holdout_predictions = _fit_predict(development, holdout, target)
        holdout_record, holdout_rows = _evaluate(holdout, holdout_predictions, target)
        result["independent_holdout"][target] = holdout_record
        prediction_rows.extend({"evaluation": "independent_holdout", **row} for row in holdout_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "strict_control_metrics.json", result)
    _write_predictions(output_dir / "strict_control_predictions.csv", prediction_rows)
    (output_dir / "strict_control_report.md").write_text(_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strict preprocessing control experiment.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    result = run_experiment(args.data_dir, args.partition, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "development_rows": result["development_rows"],
        "holdout_rows": result["holdout_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
