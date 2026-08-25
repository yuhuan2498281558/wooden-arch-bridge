from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.feature_ablation import FEATURE_SETS
from ml_pipeline.train.model_improvement import load_frozen_partition, passes_replacement_gate
from ml_pipeline.train.pilot import (
    _sha256,
    _write_csv,
    _write_json,
    applicability_domain_record,
    feature_matrix,
    group_macro_mae,
    load_training_rows,
    metric_record,
)


ARTIFACT_VERSION = "five-miao-node-pilot-v8-structure-gated-alpha"
ARTIFACT_STATUS = "research_structure_gate_not_for_deployment"
STRUCTURE_THRESHOLD = 0.5
# Half-width around 0.5 that is not a committed front/back structure type.
# Measured on 103 JSON annotations: 22 samples have |α-0.5| < 0.03; 远济
# holdout mean α=0.503 (L/R 0.506/0.500, symmetry 0.006) was a false
# back_half split under a knife-edge at 0.5. Do not magic-number this band.
AMBIGUITY_BAND = 0.03
STRUCTURE_BOUNDARY_BAND = AMBIGUITY_BAND
UNCOMMITTED_STRUCTURE_MODE = "uncommitted"
UNCOMMITTED_STRUCTURE_CLASS = -1
GATE_FEATURE_SETS = (
    "span_only",
    "span_rise_ratio",
    "span_rise_layout",
)
EXPERT_FEATURE_SETS = (
    "span_only",
    "span_rise_ratio",
)
GATE_C_VALUES = (0.1, 1.0, 10.0)
EXPERT_RECIPES: tuple[tuple[str, float | None], ...] = (
    ("median", None),
    ("ridge", 1e-8),
    ("ridge", 0.1),
    ("ridge", 1.0),
    ("ridge", 10.0),
)


def alpha_structure_class(alpha: float, threshold: float = STRUCTURE_THRESHOLD) -> int:
    """Geometric half of A-B: 0 if alpha < 0.5 else 1.

    This is the clip bound for a *committed* designer-selected half, not the
    automatic derived-mode labeler. Use :func:`derived_structure_mode` to
    decide whether a historical sample may be assigned front_half/back_half.
    """
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite")
    return int(value >= float(threshold))


def in_structure_boundary_band(
    alpha: float,
    *,
    threshold: float = STRUCTURE_THRESHOLD,
    band: float = STRUCTURE_BOUNDARY_BAND,
) -> bool:
    """Return True when |alpha - 0.5| is inside the documented boundary band."""
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite")
    return abs(value - float(threshold)) <= float(band)


def clearly_front_structure_side(
    alpha: float,
    *,
    threshold: float = STRUCTURE_THRESHOLD,
    band: float = STRUCTURE_BOUNDARY_BAND,
) -> bool:
    """True when alpha is clearly in the front half, not merely < 0.5."""
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite")
    return value < float(threshold) - float(band)


def clearly_back_structure_side(
    alpha: float,
    *,
    threshold: float = STRUCTURE_THRESHOLD,
    band: float = STRUCTURE_BOUNDARY_BAND,
) -> bool:
    """True when alpha is clearly in the back half, not merely >= 0.5."""
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite")
    return value > float(threshold) + float(band)


def derived_structure_mode(
    mean_alpha: float,
    left_alpha: float | None = None,
    right_alpha: float | None = None,
    *,
    threshold: float = STRUCTURE_THRESHOLD,
    band: float = STRUCTURE_BOUNDARY_BAND,
) -> str:
    """Commit front_half/back_half only when the sample is clearly off 0.5.

    Far from 0.5 the geometric split is unchanged: front_half if alpha < 0.5,
    back_half if alpha >= 0.5. Inside ``STRUCTURE_BOUNDARY_BAND`` (±0.03), do
    not assign back_half just because mean alpha >= 0.5.

    When left and right observations are supplied, both sides must be clearly
    on the same half before a mode is committed. Tiny L/R straddles of 0.5
    (远济 0.506/0.500) are uncommitted, not mixed structure types.
    """
    mean = float(mean_alpha)
    if not np.isfinite(mean):
        raise ValueError("alpha must be finite")
    if left_alpha is None or right_alpha is None:
        if in_structure_boundary_band(mean, threshold=threshold, band=band):
            return UNCOMMITTED_STRUCTURE_MODE
        return "back_half" if mean >= float(threshold) else "front_half"
    left = float(left_alpha)
    right = float(right_alpha)
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("alpha must be finite")
    left_front = clearly_front_structure_side(left, threshold=threshold, band=band)
    right_front = clearly_front_structure_side(right, threshold=threshold, band=band)
    left_back = clearly_back_structure_side(left, threshold=threshold, band=band)
    right_back = clearly_back_structure_side(right, threshold=threshold, band=band)
    if left_front and right_front:
        return "front_half"
    if left_back and right_back:
        return "back_half"
    return UNCOMMITTED_STRUCTURE_MODE


def derived_structure_class(
    mean_alpha: float,
    left_alpha: float | None = None,
    right_alpha: float | None = None,
    *,
    threshold: float = STRUCTURE_THRESHOLD,
    band: float = STRUCTURE_BOUNDARY_BAND,
) -> int:
    """Return 0/1 for a committed half, or -1 when the sample is uncommitted."""
    mode = derived_structure_mode(
        mean_alpha,
        left_alpha,
        right_alpha,
        threshold=threshold,
        band=band,
    )
    if mode == "front_half":
        return 0
    if mode == "back_half":
        return 1
    return UNCOMMITTED_STRUCTURE_CLASS


def left_right_structure_disagreement(
    left_alpha: float,
    right_alpha: float,
    *,
    threshold: float = STRUCTURE_THRESHOLD,
    band: float = STRUCTURE_BOUNDARY_BAND,
) -> bool:
    """True only when the two sides are clearly opposite structure types.

    A ~0.006 straddle of 0.5 is measurement noise around the boundary, not
    one-side-front / one-side-back mixed construction.
    """
    left = float(left_alpha)
    right = float(right_alpha)
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("alpha must be finite")
    left_front = clearly_front_structure_side(left, threshold=threshold, band=band)
    right_front = clearly_front_structure_side(right, threshold=threshold, band=band)
    left_back = clearly_back_structure_side(left, threshold=threshold, band=band)
    right_back = clearly_back_structure_side(right, threshold=threshold, band=band)
    return (left_front and right_back) or (left_back and right_front)


def _target(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["design_target_alpha"]) for row in rows], dtype=float)


def _classes(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray([alpha_structure_class(row["design_target_alpha"]) for row in rows], dtype=int)


def _gate(c_value: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=3000,
                random_state=20260814,
            ),
        ),
    ])


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("regressor", Ridge(alpha=float(alpha))),
    ])


def _fit_expert(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    ridge_alpha: float | None,
) -> dict[str, Any]:
    if len(y) < 3:
        raise ValueError("each structure expert requires at least three training rows")
    if model_name == "median":
        return {"model_name": model_name, "constant": float(np.median(y)), "estimator": None}
    if model_name != "ridge" or ridge_alpha is None:
        raise ValueError(f"unsupported expert recipe: {model_name}/{ridge_alpha}")
    estimator = _ridge(ridge_alpha)
    estimator.fit(x, y)
    return {
        "model_name": model_name,
        "constant": None,
        "estimator": estimator,
        "ridge_alpha": float(ridge_alpha),
    }


def _predict_expert(expert: dict[str, Any], x: np.ndarray) -> np.ndarray:
    estimator = expert.get("estimator")
    if estimator is None:
        return np.full(len(x), float(expert["constant"]), dtype=float)
    return np.asarray(estimator.predict(x), dtype=float)


def fit_structure_gated_model(
    rows: list[dict[str, Any]],
    *,
    gate_feature_set: str,
    expert_feature_set: str,
    gate_c: float,
    expert_model: str,
    expert_ridge_alpha: float | None,
) -> dict[str, Any]:
    if gate_feature_set not in GATE_FEATURE_SETS:
        raise ValueError(f"unsupported gate feature set: {gate_feature_set}")
    if expert_feature_set not in EXPERT_FEATURE_SETS:
        raise ValueError(f"unsupported expert feature set: {expert_feature_set}")
    y = _target(rows)
    classes = _classes(rows)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("structure-gated training requires both front- and back-half classes")
    gate_features = FEATURE_SETS[gate_feature_set]
    expert_features = FEATURE_SETS[expert_feature_set]
    gate_x = feature_matrix(rows, gate_features)
    expert_x = feature_matrix(rows, expert_features)
    classifier = _gate(gate_c)
    classifier.fit(gate_x, classes)
    experts = {
        class_name: _fit_expert(
            expert_x[classes == class_index],
            y[classes == class_index],
            expert_model,
            expert_ridge_alpha,
        )
        for class_index, class_name in ((0, "front_half"), (1, "back_half"))
    }
    return {
        "gate": classifier,
        "gate_feature_set": gate_feature_set,
        "gate_feature_names": list(gate_features),
        "gate_c": float(gate_c),
        "expert_feature_set": expert_feature_set,
        "expert_feature_names": list(expert_features),
        "expert_model": expert_model,
        "expert_ridge_alpha": expert_ridge_alpha,
        "experts": experts,
    }


def predict_structure_gated_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    oracle_classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_x = feature_matrix(rows, tuple(model["gate_feature_names"]))
    expert_x = feature_matrix(rows, tuple(model["expert_feature_names"]))
    probabilities = np.asarray(model["gate"].predict_proba(gate_x)[:, 1], dtype=float)
    predicted_classes = (probabilities >= 0.5).astype(int)
    routed_classes = predicted_classes if oracle_classes is None else np.asarray(oracle_classes, dtype=int)
    if len(routed_classes) != len(rows) or np.any((routed_classes < 0) | (routed_classes > 1)):
        raise ValueError("oracle classes must align with rows and contain only 0/1")
    prediction = np.zeros(len(rows), dtype=float)
    for class_index, class_name in ((0, "front_half"), (1, "back_half")):
        selected = routed_classes == class_index
        if np.any(selected):
            prediction[selected] = _predict_expert(model["experts"][class_name], expert_x[selected])
    prediction = np.where(
        routed_classes == 0,
        np.clip(prediction, 0.0, np.nextafter(STRUCTURE_THRESHOLD, 0.0)),
        np.clip(prediction, STRUCTURE_THRESHOLD, 1.0),
    )
    return prediction, probabilities, predicted_classes


def _classification_metrics(observed: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "macro_f1": float(f1_score(observed, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "confusion_matrix": [
            [int(np.sum((observed == actual) & (predicted == predicted_class))) for predicted_class in (0, 1)]
            for actual in (0, 1)
        ],
    }


def _screening_metrics(
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    probability: np.ndarray,
    oracle_prediction: np.ndarray,
) -> dict[str, Any]:
    observed = _target(rows)
    errors = np.abs(observed - prediction)
    oracle_errors = np.abs(observed - oracle_prediction)
    split_groups = np.asarray([row["split_group_key"] for row in rows])
    bridge_keys = np.asarray([row["bridge_key"] for row in rows])
    return {
        "mae": float(np.mean(errors)),
        "bridge_macro_mae": group_macro_mae(observed, prediction, bridge_keys),
        "split_group_macro_mae": group_macro_mae(observed, prediction, split_groups),
        "absolute_error_quantiles": {
            "q50": float(np.quantile(errors, 0.5)),
            "q90": float(np.quantile(errors, 0.9)),
            "q95": float(np.quantile(errors, 0.95)),
        },
        "max_absolute_error": float(np.max(errors)),
        "oracle_type_bridge_macro_mae": group_macro_mae(observed, oracle_prediction, bridge_keys),
        "oracle_type_q90": float(np.quantile(oracle_errors, 0.9)),
        "gate": _classification_metrics(_classes(rows), probability),
    }


def cross_validated_predictions(
    rows: list[dict[str, Any]],
    *,
    gate_feature_set: str,
    expert_feature_set: str,
    gate_c: float,
    expert_model: str,
    expert_ridge_alpha: float | None,
    splitter: GroupKFold | LeaveOneGroupOut,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray([row["split_group_key"] for row in rows])
    prediction = np.full(len(rows), np.nan, dtype=float)
    probability = np.full(len(rows), np.nan, dtype=float)
    oracle_prediction = np.full(len(rows), np.nan, dtype=float)
    fold_audit: list[dict[str, Any]] = []
    placeholder = np.zeros((len(rows), 1), dtype=float)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(placeholder, groups=groups), start=1):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise AssertionError("split_group_key leakage detected")
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        model = fit_structure_gated_model(
            train_rows,
            gate_feature_set=gate_feature_set,
            expert_feature_set=expert_feature_set,
            gate_c=gate_c,
            expert_model=expert_model,
            expert_ridge_alpha=expert_ridge_alpha,
        )
        fold_prediction, fold_probability, _ = predict_structure_gated_model(model, test_rows)
        fold_oracle, _, _ = predict_structure_gated_model(
            model,
            test_rows,
            oracle_classes=_classes(test_rows),
        )
        prediction[test_index] = fold_prediction
        probability[test_index] = fold_probability
        oracle_prediction[test_index] = fold_oracle
        fold_audit.append({
            "fold": fold_index,
            "training_groups": len(train_groups),
            "validation_groups": len(test_groups),
            "group_overlap": 0,
        })
    if not all(np.all(np.isfinite(values)) for values in (prediction, probability, oracle_prediction)):
        raise RuntimeError("cross-validation did not produce complete finite predictions")
    return prediction, probability, oracle_prediction, fold_audit


def _candidate_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for gate_feature_set in GATE_FEATURE_SETS:
        for expert_feature_set in EXPERT_FEATURE_SETS:
            for gate_c in GATE_C_VALUES:
                for expert_model, expert_ridge_alpha in EXPERT_RECIPES:
                    prediction, probability, oracle_prediction, _ = cross_validated_predictions(
                        rows,
                        gate_feature_set=gate_feature_set,
                        expert_feature_set=expert_feature_set,
                        gate_c=gate_c,
                        expert_model=expert_model,
                        expert_ridge_alpha=expert_ridge_alpha,
                        splitter=GroupKFold(n_splits=5),
                    )
                    record = {
                        "gate_feature_set": gate_feature_set,
                        "expert_feature_set": expert_feature_set,
                        "gate_c": gate_c,
                        "expert_model": expert_model,
                        "expert_ridge_alpha": expert_ridge_alpha,
                        "metrics": _screening_metrics(rows, prediction, probability, oracle_prediction),
                    }
                    records.append(record)
    records.sort(key=lambda item: (
        float(item["metrics"]["bridge_macro_mae"]),
        float(item["metrics"]["absolute_error_quantiles"]["q90"]),
        -float(item["metrics"]["gate"]["balanced_accuracy"]),
        len(FEATURE_SETS[item["gate_feature_set"]]),
        len(FEATURE_SETS[item["expert_feature_set"]]),
    ))
    return records


def _load_ordered_predictions(path: Path, rows: list[dict[str, Any]], field: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    by_sample = {record["sample_key"]: float(record[field]) for record in records}
    expected = {row["sample_key"] for row in rows}
    if set(by_sample) != expected:
        raise ValueError(f"baseline predictions do not match rows: {path}")
    return np.asarray([by_sample[row["sample_key"]] for row in rows], dtype=float)


def _prediction_rows(
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    probability: np.ndarray,
    oracle_prediction: np.ndarray,
) -> list[dict[str, Any]]:
    observed = _target(rows)
    observed_class = _classes(rows)
    predicted_class = (probability >= 0.5).astype(int)
    return [
        {
            "sample_key": row["sample_key"],
            "bridge_key": row["bridge_key"],
            "bridge_name": row["bridge_name"],
            "split_group_key": row["split_group_key"],
            "span_m": row["span_m"],
            "observed_alpha": observed[index],
            "observed_structure_class": "back_half" if observed_class[index] else "front_half",
            "predicted_structure_class": "back_half" if predicted_class[index] else "front_half",
            "back_half_probability": probability[index],
            "predicted_alpha": prediction[index],
            "absolute_error_alpha": abs(observed[index] - prediction[index]),
            "oracle_type_predicted_alpha": oracle_prediction[index],
            "oracle_type_absolute_error_alpha": abs(observed[index] - oracle_prediction[index]),
            "gate_uncertain": abs(probability[index] - 0.5) < 0.1,
        }
        for index, row in enumerate(rows)
    ]


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    alpha = _target(rows)
    left = np.asarray([float(row["observed_alpha_left"]) for row in rows])
    right = np.asarray([float(row["observed_alpha_right"]) for row in rows])
    classes = _classes(rows)
    return {
        "threshold": STRUCTURE_THRESHOLD,
        "definition": {
            "front_half": "design_target_alpha < 0.5",
            "back_half": "design_target_alpha >= 0.5",
            "disagreement": (
                "both sides clearly opposite halves; |α-0.5| <= "
                f"{STRUCTURE_BOUNDARY_BAND:g} is not a mixed structure type"
            ),
        },
        "rows": len(rows),
        "front_half_rows": int(np.sum(classes == 0)),
        "back_half_rows": int(np.sum(classes == 1)),
        "boundary_band": STRUCTURE_BOUNDARY_BAND,
        "within_0.03_of_threshold": int(np.sum(np.abs(alpha - STRUCTURE_THRESHOLD) <= STRUCTURE_BOUNDARY_BAND)),
        "left_right_half_disagreement_rows": int(sum(
            left_right_structure_disagreement(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )),
        "alpha_quantiles": {
            str(quantile): float(np.quantile(alpha, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
    }


def _report(
    distribution: dict[str, Any],
    selected: dict[str, Any],
    confirmation: dict[str, Any],
    baseline_confirmation: dict[str, Any],
    holdout: dict[str, Any],
    baseline_holdout: dict[str, Any],
    replacement_passed: bool,
    replacement_reasons: list[str],
) -> str:
    conclusion = (
        "自动门控达到替换门槛，可进入下一轮受控接入评审。"
        if replacement_passed
        else "类型已知时双专家明显改善，但自动门控未达到替换门槛，暂不覆盖线上 v6。"
    )
    return "\n".join([
        "# 五节苗外节点两类结构门控实验（v8）",
        "",
        f"> 状态：`{ARTIFACT_STATUS}`。{conclusion}",
        "",
        "## 结构定义与样本",
        "",
        f"- 前半区：`alpha < 0.5`，{distribution['front_half_rows']} 条；后半区：`alpha >= 0.5`，{distribution['back_half_rows']} 条（v8 自动门控训练标签仍用几何 0.5 切分）。",
        f"- 边界 ±{STRUCTURE_BOUNDARY_BAND:g} 有 {distribution['within_0.03_of_threshold']} 条；左右两侧都明显落在相反半区才计为结构分歧，现有 {distribution['left_right_half_disagreement_rows']} 条。",
        "- 分类标签只用于训练；自动预测时门控模型不能读取真实 alpha。",
        "",
        "## 选择结果",
        "",
        f"- 门控：Logistic，特征 `{selected['gate_feature_set']}`，C={selected['gate_c']}。",
        f"- 两类专家：各自独立拟合 `{selected['expert_model']}`，特征 `{selected['expert_feature_set']}`。",
        f"- 开发集 LOGO 门控平衡准确率：`{confirmation['gate']['balanced_accuracy']:.5f}`。",
        f"- 开发集 LOGO 自动门控 MAE：`{confirmation['bridge_macro_mae']:.5f}`；已知真实类型的诊断上限 MAE：`{confirmation['oracle_type_bridge_macro_mae']:.5f}`。",
        f"- v6 开发集对照 MAE：`{baseline_confirmation['bridge_macro_mae']:.5f}`。",
        f"- 独立内部留出自动门控 MAE：`{holdout['bridge_macro_mae']:.5f}`；v6 对照：`{baseline_holdout['bridge_macro_mae']:.5f}`。",
        "",
        "## 替换决定",
        "",
        f"- 是否通过：`{str(replacement_passed).lower()}`。",
        f"- 原因：`{', '.join(replacement_reasons) if replacement_reasons else 'none'}`。",
        "- 独立内部留出未参与模型或参数选择，仍不属于外部验证。",
        "",
        "## 工程解释",
        "",
        "双专家的潜在收益成立，但目前主要误差来自结构类型判错。若设计时能由用户或可靠构造字段明确选择前半区/后半区，可使用两类专家；若只能依靠现有总体参数自动判断，则应继续返回类型概率和不确定提示，不应静默替换 v6。",
        "",
    ])


def run_structure_gated_alpha(
    input_dir: Path,
    output_dir: Path,
    *,
    partition_file: Path,
    baseline_dir: Path,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260814,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    all_rows, accepted = load_training_rows(paths)
    development, holdout = load_frozen_partition(accepted, partition_file)
    candidates = _candidate_records(development)
    selected = candidates[0]
    confirmation_prediction, confirmation_probability, confirmation_oracle, fold_audit = cross_validated_predictions(
        development,
        gate_feature_set=selected["gate_feature_set"],
        expert_feature_set=selected["expert_feature_set"],
        gate_c=float(selected["gate_c"]),
        expert_model=selected["expert_model"],
        expert_ridge_alpha=selected["expert_ridge_alpha"],
        splitter=LeaveOneGroupOut(),
    )
    confirmation_metrics = metric_record(
        _target(development),
        confirmation_prediction,
        np.asarray([row["split_group_key"] for row in development]),
        bridge_keys=np.asarray([row["bridge_key"] for row in development]),
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    confirmation_metrics.update({
        "oracle_type_bridge_macro_mae": group_macro_mae(
            _target(development),
            confirmation_oracle,
            np.asarray([row["bridge_key"] for row in development]),
        ),
        "oracle_type_q90": float(np.quantile(np.abs(_target(development) - confirmation_oracle), 0.9)),
        "gate": _classification_metrics(_classes(development), confirmation_probability),
    })
    baseline_confirmation_prediction = _load_ordered_predictions(
        baseline_dir / "development_selected_oof.csv", development, "predicted_alpha"
    )
    baseline_confirmation_metrics = metric_record(
        _target(development),
        baseline_confirmation_prediction,
        np.asarray([row["split_group_key"] for row in development]),
        bridge_keys=np.asarray([row["bridge_key"] for row in development]),
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 1,
    )
    replacement_passed, replacement_reasons = passes_replacement_gate(
        confirmation_metrics,
        baseline_confirmation_metrics,
    )

    development_model = fit_structure_gated_model(
        development,
        gate_feature_set=selected["gate_feature_set"],
        expert_feature_set=selected["expert_feature_set"],
        gate_c=float(selected["gate_c"]),
        expert_model=selected["expert_model"],
        expert_ridge_alpha=selected["expert_ridge_alpha"],
    )
    holdout_prediction, holdout_probability, _ = predict_structure_gated_model(development_model, holdout)
    holdout_oracle, _, _ = predict_structure_gated_model(
        development_model, holdout, oracle_classes=_classes(holdout)
    )
    holdout_metrics = metric_record(
        _target(holdout),
        holdout_prediction,
        np.asarray([row["split_group_key"] for row in holdout]),
        bridge_keys=np.asarray([row["bridge_key"] for row in holdout]),
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 2,
    )
    holdout_metrics.update({
        "oracle_type_bridge_macro_mae": group_macro_mae(
            _target(holdout), holdout_oracle, np.asarray([row["bridge_key"] for row in holdout])
        ),
        "oracle_type_q90": float(np.quantile(np.abs(_target(holdout) - holdout_oracle), 0.9)),
        "gate": _classification_metrics(_classes(holdout), holdout_probability),
    })
    baseline_holdout_prediction = _load_ordered_predictions(
        baseline_dir / "independent_holdout_predictions.csv", holdout, "predicted_alpha"
    )
    baseline_holdout_metrics = metric_record(
        _target(holdout),
        baseline_holdout_prediction,
        np.asarray([row["split_group_key"] for row in holdout]),
        bridge_keys=np.asarray([row["bridge_key"] for row in holdout]),
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 3,
    )

    full_model = fit_structure_gated_model(
        accepted,
        gate_feature_set=selected["gate_feature_set"],
        expert_feature_set=selected["expert_feature_set"],
        gate_c=float(selected["gate_c"]),
        expert_model=selected["expert_model"],
        expert_ridge_alpha=selected["expert_ridge_alpha"],
    )
    all_gate_features = feature_matrix(accepted, tuple(full_model["gate_feature_names"]))
    full_model.update({
        "artifact_version": ARTIFACT_VERSION,
        "status": ARTIFACT_STATUS,
        "target": "alpha",
        "target_column": "design_target_alpha",
        "structure_threshold": STRUCTURE_THRESHOLD,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(accepted),
        "training_groups": len({row["split_group_key"] for row in accepted}),
        "training_feature_range": {
            name: [float(np.min(all_gate_features[:, index])), float(np.max(all_gate_features[:, index]))]
            for index, name in enumerate(full_model["gate_feature_names"])
        },
        "applicability_domain": applicability_domain_record(
            all_gate_features, tuple(full_model["gate_feature_names"])
        ),
        "development_confirmation_metrics": confirmation_metrics,
        "independent_holdout_metrics": holdout_metrics,
        "replacement_gate_passed": replacement_passed,
        "replacement_gate_reasons": replacement_reasons,
    })
    artifact_dir = output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_model, artifact_dir / "alpha_structure_gated_model.joblib")

    distribution = _distribution(accepted)
    selected_record = {
        key: value for key, value in selected.items() if key != "metrics"
    }
    selected_record["screening_metrics"] = selected["metrics"]
    selected_record["replacement_gate_passed"] = replacement_passed
    selected_record["replacement_gate_reasons"] = replacement_reasons
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "status": ARTIFACT_STATUS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "partition_source": str(partition_file),
        "partition_sha256": _sha256(partition_file),
        "baseline_directory": str(baseline_dir),
        "source_files": [{"name": path.name, "sha256": _sha256(path)} for path in paths],
        "source_rows": len(all_rows),
        "training_rows": len(accepted),
        "development_rows": len(development),
        "independent_holdout_rows": len(holdout),
        "structure_distribution": distribution,
        "selected": selected_record,
        "validation": {
            "screening_outer": "GroupKFold(5)",
            "confirmation_outer": "LeaveOneGroupOut",
            "split_by": "split_group_key",
            "score_by": "bridge_key",
            "holdout_used_for_selection": False,
            "target_leakage_guard": "gate never receives observed/design_target alpha at prediction time",
        },
        "random_seed": random_seed,
        "bootstrap_iterations": bootstrap_iterations,
    }
    _write_json(output_dir / "structure_distribution.json", distribution)
    _write_json(output_dir / "screening_candidates.json", candidates)
    _write_json(output_dir / "selected_model.json", selected_record)
    _write_json(output_dir / "confirmation_metrics.json", confirmation_metrics)
    _write_json(output_dir / "baseline_confirmation_metrics.json", baseline_confirmation_metrics)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_json(output_dir / "baseline_independent_holdout_metrics.json", baseline_holdout_metrics)
    _write_json(output_dir / "fold_audit.json", fold_audit)
    _write_json(output_dir / "training_manifest.json", manifest)
    _write_csv(
        output_dir / "development_predictions.csv",
        _prediction_rows(development, confirmation_prediction, confirmation_probability, confirmation_oracle),
    )
    _write_csv(
        output_dir / "independent_holdout_predictions.csv",
        _prediction_rows(holdout, holdout_prediction, holdout_probability, holdout_oracle),
    )
    (output_dir / "structure_gated_alpha_report.md").write_text(
        _report(
            distribution,
            selected_record,
            confirmation_metrics,
            baseline_confirmation_metrics,
            holdout_metrics,
            baseline_holdout_metrics,
            replacement_passed,
            replacement_reasons,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "selected": selected_record,
        "replacement_gate_passed": replacement_passed,
        "replacement_gate_reasons": replacement_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a two-class gate plus alpha experts.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260814)
    args = parser.parse_args()
    result = run_structure_gated_alpha(
        args.input_dir,
        args.output_dir,
        partition_file=args.partition_file,
        baseline_dir=args.baseline_dir,
        pattern=args.pattern,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
