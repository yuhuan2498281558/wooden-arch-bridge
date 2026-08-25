from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.feature_ablation import FEATURE_SETS
from ml_pipeline.train.model_improvement import load_frozen_partition
from ml_pipeline.train.pilot import (
    MODEL_SELECTION_TOLERANCE,
    _sha256,
    _write_csv,
    _write_json,
    applicability_domain_record,
    feature_matrix,
    group_macro_mae,
    load_training_rows,
    make_estimator,
    metric_record,
)
from ml_pipeline.train.structure_gated_alpha import (
    STRUCTURE_BOUNDARY_BAND,
    STRUCTURE_THRESHOLD,
    UNCOMMITTED_STRUCTURE_CLASS,
    UNCOMMITTED_STRUCTURE_MODE,
    derived_structure_class,
    derived_structure_mode,
    in_structure_boundary_band,
    left_right_structure_disagreement,
)


ARTIFACT_VERSION = "five-miao-node-pilot-v9-designer-selected-mode"
ARTIFACT_STATUS = "research_designer_mode_not_for_deployment"
MODE_NAMES = ("front_half", "back_half")
EXPERT_FEATURE_SETS = ("span_only", "span_rise_ratio")
RIDGE_ALPHAS = (1e-8, 0.1, 1.0, 10.0)
HUBER_RECIPES = (
    {"alpha": 1e-4, "epsilon": 1.1},
    {"alpha": 1e-4, "epsilon": 1.35},
    {"alpha": 1e-2, "epsilon": 1.35},
    {"alpha": 0.1, "epsilon": 1.35},
)
TAIL_ERROR_TOLERANCE = 0.005


def _target(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["design_target_alpha"]) for row in rows], dtype=float)


def _row_left_right(row: dict[str, Any]) -> tuple[float | None, float | None]:
    left = row.get("observed_alpha_left")
    right = row.get("observed_alpha_right")
    if left is None or right is None:
        return None, None
    return float(left), float(right)


def _derived_mode_name(row: dict[str, Any]) -> str:
    left, right = _row_left_right(row)
    return derived_structure_mode(row["design_target_alpha"], left, right)


def _classes(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            derived_structure_class(row["design_target_alpha"], *_row_left_right(row))
            for row in rows
        ],
        dtype=int,
    )


def _candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{
        "model_name": "median",
        "feature_set": "none",
        "feature_names": [],
        "params": {},
    }]
    for feature_set in EXPERT_FEATURE_SETS:
        feature_names = list(FEATURE_SETS[feature_set])
        for alpha in RIDGE_ALPHAS:
            specs.append({
                "model_name": "ridge",
                "feature_set": feature_set,
                "feature_names": feature_names,
                "params": {"alpha": alpha},
            })
        for params in HUBER_RECIPES:
            specs.append({
                "model_name": "huber",
                "feature_set": feature_set,
                "feature_names": feature_names,
                "params": dict(params),
            })
    return specs


def _fit_expert(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < 3:
        raise ValueError("each design-mode expert requires at least three training rows")
    target = _target(rows)
    model_name = str(spec["model_name"])
    if model_name == "median":
        return {
            "model_name": model_name,
            "feature_set": "none",
            "feature_names": [],
            "params": {},
            "constant": float(np.median(target)),
            "estimator": None,
        }
    feature_names = tuple(spec["feature_names"])
    estimator = make_estimator(model_name, dict(spec["params"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(feature_matrix(rows, feature_names), target)
    return {
        "model_name": model_name,
        "feature_set": spec["feature_set"],
        "feature_names": list(feature_names),
        "params": dict(spec["params"]),
        "constant": None,
        "estimator": estimator,
    }


def _predict_expert(expert: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    estimator = expert.get("estimator")
    if estimator is None:
        return np.full(len(rows), float(expert["constant"]), dtype=float)
    names = tuple(expert["feature_names"])
    return np.asarray(estimator.predict(feature_matrix(rows, names)), dtype=float)


def _ungated_expert_mix(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    """Boundary-band prediction: interpolate the two half-experts without a 0.5 clip."""
    front = _predict_expert(model["experts"]["front_half"], rows)
    back = _predict_expert(model["experts"]["back_half"], rows)
    return 0.5 * (front + back)


def clip_to_design_mode(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    predictions = np.asarray(values, dtype=float)
    modes = np.asarray(classes, dtype=int)
    if len(predictions) != len(modes) or np.any((modes < -1) | (modes > 1)):
        raise ValueError("design modes must align with predictions and contain only -1/0/1")
    lower_open = np.nextafter(0.0, 1.0)
    front_upper = np.nextafter(STRUCTURE_THRESHOLD, 0.0)
    upper_open = np.nextafter(1.0, 0.0)
    return np.where(
        modes == 0,
        np.clip(predictions, lower_open, front_upper),
        np.where(
            modes == 1,
            np.clip(predictions, STRUCTURE_THRESHOLD, upper_open),
            np.clip(predictions, lower_open, upper_open),
        ),
    )


def _normalize_modes(modes: Iterable[str | int], expected_length: int) -> np.ndarray:
    mapping = {
        "front_half": 0,
        "back_half": 1,
        UNCOMMITTED_STRUCTURE_MODE: UNCOMMITTED_STRUCTURE_CLASS,
        0: 0,
        1: 1,
        -1: UNCOMMITTED_STRUCTURE_CLASS,
    }
    try:
        normalized = np.asarray([mapping[value] for value in modes], dtype=int)
    except (KeyError, TypeError) as exc:
        raise ValueError("design mode must be front_half/back_half/uncommitted or 0/1/-1") from exc
    if len(normalized) != expected_length:
        raise ValueError("design modes must align with rows")
    return normalized


def fit_design_mode_model(
    rows: list[dict[str, Any]],
    selected_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    classes = _classes(rows)
    committed = classes >= 0
    if set(classes[committed].tolist()) != {0, 1}:
        raise ValueError("design-mode training requires both front- and back-half rows")
    experts: dict[str, dict[str, Any]] = {}
    for class_index, mode_name in enumerate(MODE_NAMES):
        mode_rows = [row for index, row in enumerate(rows) if classes[index] == class_index]
        experts[mode_name] = _fit_expert(mode_rows, selected_specs[mode_name])
    return {
        "experts": experts,
        "structure_threshold": STRUCTURE_THRESHOLD,
        "structure_boundary_band": STRUCTURE_BOUNDARY_BAND,
        "mode_required": True,
    }


def predict_design_mode_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    modes: Iterable[str | int],
    *,
    ungated_fallback: np.ndarray | None = None,
) -> np.ndarray:
    classes = _normalize_modes(modes, len(rows))
    predictions = np.zeros(len(rows), dtype=float)
    for class_index, mode_name in enumerate(MODE_NAMES):
        selected = np.flatnonzero(classes == class_index)
        if len(selected):
            mode_rows = [rows[index] for index in selected]
            predictions[selected] = _predict_expert(model["experts"][mode_name], mode_rows)
    uncommitted = np.flatnonzero(classes < 0)
    if len(uncommitted):
        if ungated_fallback is not None:
            fallback = np.asarray(ungated_fallback, dtype=float)
            if len(fallback) != len(rows):
                raise ValueError("ungated fallback must align with rows")
            predictions[uncommitted] = fallback[uncommitted]
        else:
            uncommitted_rows = [rows[index] for index in uncommitted]
            predictions[uncommitted] = _ungated_expert_mix(model, uncommitted_rows)
    return clip_to_design_mode(predictions, classes)


def _split_indices(
    rows: list[dict[str, Any]],
    splitter: GroupKFold | LeaveOneGroupOut,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([row["split_group_key"] for row in rows])
    placeholder = np.zeros((len(rows), 1), dtype=float)
    for train_index, test_index in splitter.split(placeholder, groups=groups):
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError("split_group_key leakage detected")
        yield train_index, test_index


def cross_validated_predictions(
    rows: list[dict[str, Any]],
    selected_specs: dict[str, dict[str, Any]],
    *,
    splitter: GroupKFold | LeaveOneGroupOut,
    ungated_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    classes = _classes(rows)
    predictions = np.full(len(rows), np.nan, dtype=float)
    fold_audit: list[dict[str, Any]] = []
    groups = np.asarray([row["split_group_key"] for row in rows])
    fallback = None if ungated_fallback is None else np.asarray(ungated_fallback, dtype=float)
    if fallback is not None and len(fallback) != len(rows):
        raise ValueError("ungated fallback must align with rows")
    for fold_index, (train_index, test_index) in enumerate(_split_indices(rows, splitter), start=1):
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        model = fit_design_mode_model(train_rows, selected_specs)
        fold_fallback = None if fallback is None else fallback[test_index]
        predictions[test_index] = predict_design_mode_model(
            model,
            test_rows,
            classes[test_index],
            ungated_fallback=fold_fallback,
        )
        fold_audit.append({
            "fold": fold_index,
            "training_groups": len(set(groups[train_index])),
            "validation_groups": len(set(groups[test_index])),
            "group_overlap": 0,
            "training_front_half_rows": int(np.sum(classes[train_index] == 0)),
            "training_back_half_rows": int(np.sum(classes[train_index] == 1)),
            "training_uncommitted_rows": int(np.sum(classes[train_index] < 0)),
            "validation_front_half_rows": int(np.sum(classes[test_index] == 0)),
            "validation_back_half_rows": int(np.sum(classes[test_index] == 1)),
            "validation_uncommitted_rows": int(np.sum(classes[test_index] < 0)),
        })
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("cross-validation did not produce complete finite predictions")
    return predictions, fold_audit


def _screen_candidate(
    rows: list[dict[str, Any]],
    mode_index: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    classes = _classes(rows)
    selected_indices = np.flatnonzero(classes == mode_index)
    predictions = np.full(len(rows), np.nan, dtype=float)
    try:
        for train_index, test_index in _split_indices(rows, GroupKFold(n_splits=5)):
            mode_train = [rows[index] for index in train_index if classes[index] == mode_index]
            mode_test_indices = np.asarray(
                [index for index in test_index if classes[index] == mode_index],
                dtype=int,
            )
            if not len(mode_test_indices):
                continue
            expert = _fit_expert(mode_train, spec)
            mode_test = [rows[index] for index in mode_test_indices]
            raw = _predict_expert(expert, mode_test)
            predictions[mode_test_indices] = clip_to_design_mode(
                raw,
                np.full(len(raw), mode_index, dtype=int),
            )
    except (ValueError, FloatingPointError) as exc:
        return {**spec, "valid": False, "failure": str(exc)}
    if not len(selected_indices):
        return {**spec, "valid": False, "failure": "no_committed_mode_rows"}
    if not np.all(np.isfinite(predictions[selected_indices])):
        return {**spec, "valid": False, "failure": "incomplete_predictions"}
    mode_rows = [rows[index] for index in selected_indices]
    observed = _target(mode_rows)
    predicted = predictions[selected_indices]
    bridge_keys = np.asarray([row["bridge_key"] for row in mode_rows])
    errors = np.abs(observed - predicted)
    return {
        **spec,
        "valid": True,
        "bridge_macro_mae": group_macro_mae(observed, predicted, bridge_keys),
        "mae": float(np.mean(errors)),
        "q90": float(np.quantile(errors, 0.9)),
        "max_absolute_error": float(np.max(errors)),
    }


def _select_spec(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid")]
    if not valid:
        raise RuntimeError("no valid design-mode expert candidates")
    best_score = min(float(record["bridge_macro_mae"]) for record in valid)
    eligible = [
        record for record in valid
        if float(record["bridge_macro_mae"]) <= best_score + MODEL_SELECTION_TOLERANCE
    ]
    simplicity = {"median": 0, "ridge": 1, "huber": 2}
    eligible.sort(key=lambda record: (
        simplicity[str(record["model_name"])],
        len(record["feature_names"]),
        float(record["bridge_macro_mae"]),
        float(record["q90"]),
    ))
    return {
        key: value for key, value in eligible[0].items()
        if key not in {"valid", "failure", "bridge_macro_mae", "mae", "q90", "max_absolute_error"}
    }


def _screen_specs(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, dict[str, Any]] = {}
    specs = _candidate_specs()
    for mode_index, mode_name in enumerate(MODE_NAMES):
        mode_records = [_screen_candidate(rows, mode_index, spec) for spec in specs]
        records[mode_name] = mode_records
        selected[mode_name] = _select_spec(mode_records)
    return selected, records


def _metrics(
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    observed = _target(rows)
    split_groups = np.asarray([row["split_group_key"] for row in rows])
    bridge_keys = np.asarray([row["bridge_key"] for row in rows])
    classes = _classes(rows)
    result = metric_record(
        observed,
        predictions,
        split_groups,
        bridge_keys=bridge_keys,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    by_mode: dict[str, dict[str, Any]] = {}
    for mode_index, mode_name in enumerate(MODE_NAMES):
        selected = classes == mode_index
        if not np.any(selected):
            raise ValueError(f"metrics require committed {mode_name} rows")
        mode_rows = [row for index, row in enumerate(rows) if selected[index]]
        by_mode[mode_name] = metric_record(
            observed[selected],
            predictions[selected],
            np.asarray([row["split_group_key"] for row in mode_rows]),
            bridge_keys=np.asarray([row["bridge_key"] for row in mode_rows]),
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + mode_index + 1,
        )
        by_mode[mode_name]["rows"] = int(np.sum(selected))
        by_mode[mode_name]["groups"] = len({row["split_group_key"] for row in mode_rows})
    result["by_mode"] = by_mode
    result["uncommitted_rows"] = int(np.sum(classes < 0))
    result["mode_macro_bridge_mae"] = float(np.mean([
        by_mode[name]["bridge_macro_mae"] for name in MODE_NAMES
    ]))
    result["worst_mode_q90"] = float(max(
        by_mode[name]["absolute_error_quantiles"]["q90"] for name in MODE_NAMES
    ))
    return result


def _load_ordered_predictions(path: Path, rows: list[dict[str, Any]], field: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    by_sample = {record["sample_key"]: float(record[field]) for record in records}
    expected = {row["sample_key"] for row in rows}
    if set(by_sample) != expected:
        raise ValueError(f"baseline predictions do not match rows: {path}")
    return np.asarray([by_sample[row["sample_key"]] for row in rows], dtype=float)


def _baseline_predictions(
    rows: list[dict[str, Any]],
    raw_v6: np.ndarray,
    mode_median: np.ndarray,
) -> dict[str, np.ndarray]:
    classes = _classes(rows)
    return {
        "fixed_rule": np.full(len(rows), 2.0 / 3.0, dtype=float),
        "v6_single_model": np.asarray(raw_v6, dtype=float),
        "v6_mode_constrained": clip_to_design_mode(raw_v6, classes),
        "mode_train_median": np.asarray(mode_median, dtype=float),
    }


def _metric_bundle(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    return {
        name: _metrics(
            rows,
            values,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + index * 10,
        )
        for index, (name, values) in enumerate(predictions.items())
    }


def _passes_integration_gate(
    candidate: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> tuple[bool, list[str], str]:
    comparator_name = min(
        baselines,
        key=lambda name: float(baselines[name]["mode_macro_bridge_mae"]),
    )
    comparator = baselines[comparator_name]
    reasons: list[str] = []
    if float(candidate["mode_macro_bridge_mae"]) > (
        float(comparator["mode_macro_bridge_mae"]) - MODEL_SELECTION_TOLERANCE
    ):
        reasons.append("mode_macro_mae_not_materially_better_than_strongest_baseline")
    if float(candidate["bridge_macro_mae"]) > (
        float(comparator["bridge_macro_mae"]) - MODEL_SELECTION_TOLERANCE
    ):
        reasons.append("overall_bridge_macro_mae_not_materially_better")
    if float(candidate["absolute_error_quantiles"]["q90"]) > (
        float(comparator["absolute_error_quantiles"]["q90"]) + TAIL_ERROR_TOLERANCE
    ):
        reasons.append("overall_q90_tail_error_worse")
    for mode_name in MODE_NAMES:
        strongest_name = min(
            baselines,
            key=lambda name: float(baselines[name]["by_mode"][mode_name]["bridge_macro_mae"]),
        )
        strongest = baselines[strongest_name]["by_mode"][mode_name]
        current = candidate["by_mode"][mode_name]
        if float(current["bridge_macro_mae"]) > (
            float(strongest["bridge_macro_mae"]) + MODEL_SELECTION_TOLERANCE
        ):
            reasons.append(f"{mode_name}_mae_worse_than_strongest_baseline")
        if float(current["absolute_error_quantiles"]["q90"]) > (
            float(strongest["absolute_error_quantiles"]["q90"]) + TAIL_ERROR_TOLERANCE
        ):
            reasons.append(f"{mode_name}_q90_worse_than_strongest_baseline")
    return not reasons, reasons, comparator_name


def _row_flags(row: dict[str, Any]) -> tuple[bool, bool]:
    alpha = float(row["design_target_alpha"])
    left, right = _row_left_right(row)
    boundary = in_structure_boundary_band(alpha)
    if left is None or right is None:
        disagreement = False
    else:
        disagreement = left_right_structure_disagreement(left, right)
    return boundary, disagreement


def _sensitivity_analysis(
    rows: list[dict[str, Any]],
    selected_specs: dict[str, dict[str, Any]],
    raw_v6_by_sample: dict[str, float],
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    filters = {
        "all_rows": lambda boundary, disagreement: True,
        "exclude_boundary_band": lambda boundary, disagreement: not boundary,
        "exclude_left_right_disagreement": lambda boundary, disagreement: not disagreement,
        "exclude_boundary_or_disagreement": lambda boundary, disagreement: not boundary and not disagreement,
    }
    result: dict[str, Any] = {}
    median_specs = {
        mode_name: {
            "model_name": "median",
            "feature_set": "none",
            "feature_names": [],
            "params": {},
        }
        for mode_name in MODE_NAMES
    }
    for index, (name, include) in enumerate(filters.items()):
        subset = [row for row in rows if include(*_row_flags(row))]
        distribution = _distribution(subset)
        if min(distribution["front_half_rows"], distribution["back_half_rows"]) < 3:
            result[name] = {"status": "insufficient_rows", "distribution": distribution}
            continue
        candidate, _ = cross_validated_predictions(
            subset,
            selected_specs,
            splitter=LeaveOneGroupOut(),
        )
        median, _ = cross_validated_predictions(
            subset,
            median_specs,
            splitter=LeaveOneGroupOut(),
        )
        raw_v6 = np.asarray([raw_v6_by_sample[row["sample_key"]] for row in subset], dtype=float)
        baselines = _baseline_predictions(subset, raw_v6, median)
        baseline_metrics = _metric_bundle(
            subset,
            baselines,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + index * 100,
        )
        candidate_metrics = _metrics(
            subset,
            candidate,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + index * 100 + 50,
        )
        passed, reasons, comparator = _passes_integration_gate(candidate_metrics, baseline_metrics)
        result[name] = {
            "status": "evaluated",
            "distribution": distribution,
            "candidate": candidate_metrics,
            "baselines": baseline_metrics,
            "integration_gate_passed": passed,
            "integration_gate_reasons": reasons,
            "strongest_overall_baseline": comparator,
        }
    return result


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = _classes(rows)
    flags = [_row_flags(row) for row in rows]
    return {
        "rows": len(rows),
        "groups": len({row["split_group_key"] for row in rows}),
        "front_half_rows": int(np.sum(classes == 0)),
        "back_half_rows": int(np.sum(classes == 1)),
        "uncommitted_rows": int(np.sum(classes < 0)),
        "boundary_band": STRUCTURE_BOUNDARY_BAND,
        "within_boundary_band_rows": sum(boundary for boundary, _ in flags),
        "within_0.03_of_threshold": sum(boundary for boundary, _ in flags),
        "left_right_half_disagreement_rows": sum(disagreement for _, disagreement in flags),
        "threshold": STRUCTURE_THRESHOLD,
        "historical_mode_definition": {
            "front_half": (
                f"both observed sides clearly below {STRUCTURE_THRESHOLD:g} - "
                f"{STRUCTURE_BOUNDARY_BAND:g}"
            ),
            "back_half": (
                f"both observed sides clearly above {STRUCTURE_THRESHOLD:g} + "
                f"{STRUCTURE_BOUNDARY_BAND:g}"
            ),
            "uncommitted": (
                f"|mean α - {STRUCTURE_THRESHOLD:g}| <= {STRUCTURE_BOUNDARY_BAND:g} "
                "or the two sides are not both clearly the same half"
            ),
            "note": (
                "远济 holdout mean α=0.503 (L/R 0.506/0.500) and 22/103 samples "
                f"with |α-0.5| <= {STRUCTURE_BOUNDARY_BAND:g} must not be labeled back_half "
                "by a knife-edge at 0.5."
            ),
        },
    }


def _prediction_rows(
    rows: list[dict[str, Any]],
    candidate: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    observed = _target(rows)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        boundary, disagreement = _row_flags(row)
        record = {
            "sample_key": row["sample_key"],
            "bridge_key": row["bridge_key"],
            "bridge_name": row["bridge_name"],
            "split_group_key": row["split_group_key"],
            "span_m": row["span_m"],
            "observed_alpha": observed[index],
            "derived_design_mode": _derived_mode_name(row),
            "within_boundary_band": boundary,
            "left_right_half_disagreement": disagreement,
            "predicted_alpha": candidate[index],
            "absolute_error_alpha": abs(observed[index] - candidate[index]),
        }
        for name, values in baselines.items():
            record[f"{name}_predicted_alpha"] = values[index]
            record[f"{name}_absolute_error_alpha"] = abs(observed[index] - values[index])
        output.append(record)
    return output


def _report(
    distribution: dict[str, Any],
    selected_specs: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    gate_passed: bool,
    gate_reasons: list[str],
    comparator_name: str,
    holdout_candidate: dict[str, Any],
    sensitivity: dict[str, Any],
) -> str:
    decision = (
        "开发集条件双专家通过预设接入门槛，可进入系统接入评审。"
        if gate_passed
        else "开发集条件双专家未通过预设接入门槛，保持研究产物，不接入默认预测。"
    )
    lines = [
        "# 五节苗设计模式条件双专家实验（v9）",
        "",
        f"> 状态：`{ARTIFACT_STATUS}`。{decision}",
        "",
        "## 研究口径",
        "",
        f"- 历史前/后半区只在两侧都明显离开 0.5±{STRUCTURE_BOUNDARY_BAND:g} 时派生；贴边样本标为 uncommitted，不因均值 ≥0.5 就切到后半区。依据：远济 holdout 均值 0.503（左右 0.506/0.500）及 22/103 条 |α-0.5|≤{STRUCTURE_BOUNDARY_BAND:g}。",
        "- 在线使用前提仍是设计人员预先给定结构模式；本实验不运行自动分类器。贴边样本的派生评估走连续/v6 式预测，不把 Ridge 专家换成别的学习器。",
        f"- 共 {distribution['rows']} 条：前半区 {distribution['front_half_rows']} 条，后半区 {distribution['back_half_rows']} 条，未提交 {distribution['uncommitted_rows']} 条。",
        f"- 边界 ±{STRUCTURE_BOUNDARY_BAND:g} 有 {distribution['within_boundary_band_rows']} 条；左右两侧都明显相反半区才计结构分歧，现有 {distribution['left_right_half_disagreement_rows']} 条。",
        "",
        "## 模型选择",
        "",
        f"- 前半区：`{selected_specs['front_half']['model_name']}` / `{selected_specs['front_half']['feature_set']}` / `{selected_specs['front_half']['params']}`。",
        f"- 后半区：`{selected_specs['back_half']['model_name']}` / `{selected_specs['back_half']['feature_set']}` / `{selected_specs['back_half']['params']}`。",
        "- beta 不在本实验中重训，系统接入时继续使用 v6 beta。",
        "",
        "## 开发集分组验证",
        "",
        f"- 条件双专家桥级宏 MAE：`{candidate['bridge_macro_mae']:.5f}`，模式宏 MAE：`{candidate['mode_macro_bridge_mae']:.5f}`，q90：`{candidate['absolute_error_quantiles']['q90']:.5f}`。",
    ]
    for name, metrics in baselines.items():
        lines.append(
            f"- `{name}`：桥级宏 MAE `{metrics['bridge_macro_mae']:.5f}`，模式宏 MAE `{metrics['mode_macro_bridge_mae']:.5f}`，q90 `{metrics['absolute_error_quantiles']['q90']:.5f}`。"
        )
    lines.extend([
        "",
        "## 接入决定",
        "",
        f"- 是否通过：`{str(gate_passed).lower()}`；最强模式宏基线：`{comparator_name}`。",
        f"- 原因：`{', '.join(gate_reasons) if gate_reasons else 'none'}`。",
        f"- 已查看的内部留出条件双专家桥级宏 MAE：`{holdout_candidate['bridge_macro_mae']:.5f}`；该留出不再视为全新外部验证。",
        "",
        "## 敏感性分析",
        "",
    ])
    for name, record in sensitivity.items():
        if record["status"] != "evaluated":
            lines.append(f"- `{name}`：样本不足。")
            continue
        lines.append(
            f"- `{name}`：{record['distribution']['rows']} 条，模式宏 MAE `{record['candidate']['mode_macro_bridge_mae']:.5f}`，接入门槛 `{str(record['integration_gate_passed']).lower()}`。"
        )
    lines.extend([
        "",
        "## 论文与工程边界",
        "",
        "该结果只能解释为设计模式给定条件下的位置区间预测。目标派生类型带来的区间信息已通过分区中位数和区间约束 v6 单独对照；不得把本实验表述为系统自动识别两种结构。",
        "",
    ])
    return "\n".join(lines)


def run_design_mode_alpha(
    input_dir: Path,
    output_dir: Path,
    *,
    partition_file: Path,
    baseline_dir: Path,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260815,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    all_rows, accepted = load_training_rows(paths)
    development, holdout = load_frozen_partition(accepted, partition_file)

    selected_specs, screening_records = _screen_specs(development)
    median_specs = {
        mode_name: {
            "model_name": "median",
            "feature_set": "none",
            "feature_names": [],
            "params": {},
        }
        for mode_name in MODE_NAMES
    }
    development_v6 = _load_ordered_predictions(
        baseline_dir / "development_selected_oof.csv",
        development,
        "predicted_alpha",
    )
    development_prediction, fold_audit = cross_validated_predictions(
        development,
        selected_specs,
        splitter=LeaveOneGroupOut(),
        ungated_fallback=development_v6,
    )
    development_median, _ = cross_validated_predictions(
        development,
        median_specs,
        splitter=LeaveOneGroupOut(),
        ungated_fallback=development_v6,
    )
    development_baseline_predictions = _baseline_predictions(
        development,
        development_v6,
        development_median,
    )
    development_metrics = _metrics(
        development,
        development_prediction,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    development_baseline_metrics = _metric_bundle(
        development,
        development_baseline_predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 100,
    )
    gate_passed, gate_reasons, comparator_name = _passes_integration_gate(
        development_metrics,
        development_baseline_metrics,
    )

    development_model = fit_design_mode_model(development, selected_specs)
    holdout_v6 = _load_ordered_predictions(
        baseline_dir / "independent_holdout_predictions.csv",
        holdout,
        "predicted_alpha",
    )
    holdout_prediction = predict_design_mode_model(
        development_model,
        holdout,
        _classes(holdout),
        ungated_fallback=holdout_v6,
    )
    development_median_model = fit_design_mode_model(development, median_specs)
    holdout_median = predict_design_mode_model(
        development_median_model,
        holdout,
        _classes(holdout),
        ungated_fallback=holdout_v6,
    )
    holdout_baseline_predictions = _baseline_predictions(holdout, holdout_v6, holdout_median)
    holdout_metrics = _metrics(
        holdout,
        holdout_prediction,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 200,
    )
    holdout_baseline_metrics = _metric_bundle(
        holdout,
        holdout_baseline_predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 300,
    )

    development_v6_by_sample = {
        row["sample_key"]: float(development_v6[index])
        for index, row in enumerate(development)
    }
    sensitivity = _sensitivity_analysis(
        development,
        selected_specs,
        development_v6_by_sample,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed + 400,
    )

    full_model = fit_design_mode_model(accepted, selected_specs)
    classes = _classes(accepted)
    for mode_index, mode_name in enumerate(MODE_NAMES):
        mode_rows = [row for index, row in enumerate(accepted) if classes[index] == mode_index]
        expert = full_model["experts"][mode_name]
        names = tuple(expert["feature_names"])
        expert.update({
            "mode": mode_name,
            "output_bounds": [0.0, STRUCTURE_THRESHOLD] if mode_index == 0 else [STRUCTURE_THRESHOLD, 1.0],
            "training_rows": len(mode_rows),
            "training_groups": len({row["split_group_key"] for row in mode_rows}),
            "training_feature_range": {
                name: [
                    float(np.min(feature_matrix(mode_rows, names)[:, feature_index])),
                    float(np.max(feature_matrix(mode_rows, names)[:, feature_index])),
                ]
                for feature_index, name in enumerate(names)
            } if names else {},
            "applicability_domain": applicability_domain_record(
                feature_matrix(mode_rows, names), names
            ) if names else None,
        })
    trained_at = datetime.now(timezone.utc).isoformat()
    full_model.update({
        "artifact_version": ARTIFACT_VERSION,
        "status": ARTIFACT_STATUS,
        "trained_at": trained_at,
        "target": "alpha",
        "target_column": "design_target_alpha",
        "historical_mode_source": "derived_from_design_target_alpha_with_boundary_band",
        "deployment_mode_source_required": "designer_selected",
        "training_rows": len(accepted),
        "training_groups": len({row["split_group_key"] for row in accepted}),
        "development_metrics": development_metrics,
        "independent_holdout_metrics": holdout_metrics,
        "integration_gate_passed": gate_passed,
        "integration_gate_reasons": gate_reasons,
    })
    artifact_dir = output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_model, artifact_dir / "alpha_design_mode_model.joblib")

    distribution = _distribution(accepted)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "status": ARTIFACT_STATUS,
        "trained_at": trained_at,
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
        "distribution": distribution,
        "selected_specs": selected_specs,
        "validation": {
            "screening_outer": "GroupKFold(5)",
            "confirmation_outer": "LeaveOneGroupOut",
            "split_by": "split_group_key",
            "score_by": "bridge_key and equal-weight design-mode macro",
            "holdout_used_for_selection": False,
            "historical_mode_source": (
                f"committed halves require both sides clearly off 0.5±{STRUCTURE_BOUNDARY_BAND:g}; "
                "uncommitted samples keep ungated/v6-style prediction"
            ),
            "deployment_contract": "designer supplies front_half/back_half before prediction",
        },
        "integration_gate_passed": gate_passed,
        "integration_gate_reasons": gate_reasons,
        "strongest_overall_baseline": comparator_name,
        "random_seed": random_seed,
        "bootstrap_iterations": bootstrap_iterations,
    }
    _write_json(output_dir / "structure_distribution.json", distribution)
    _write_json(output_dir / "screening_candidates.json", screening_records)
    _write_json(output_dir / "selected_model.json", selected_specs)
    _write_json(output_dir / "development_metrics.json", development_metrics)
    _write_json(output_dir / "development_baseline_metrics.json", development_baseline_metrics)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_json(output_dir / "independent_holdout_baseline_metrics.json", holdout_baseline_metrics)
    _write_json(output_dir / "sensitivity_analysis.json", sensitivity)
    _write_json(output_dir / "fold_audit.json", fold_audit)
    _write_json(output_dir / "training_manifest.json", manifest)
    _write_csv(
        output_dir / "development_predictions.csv",
        _prediction_rows(development, development_prediction, development_baseline_predictions),
    )
    _write_csv(
        output_dir / "independent_holdout_predictions.csv",
        _prediction_rows(holdout, holdout_prediction, holdout_baseline_predictions),
    )
    (output_dir / "design_mode_alpha_report.md").write_text(
        _report(
            distribution,
            selected_specs,
            development_metrics,
            development_baseline_metrics,
            gate_passed,
            gate_reasons,
            comparator_name,
            holdout_metrics,
            sensitivity,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "selected_specs": selected_specs,
        "integration_gate_passed": gate_passed,
        "integration_gate_reasons": gate_reasons,
        "strongest_overall_baseline": comparator_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate designer-selected front/back alpha experts without automatic gating."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260815)
    args = parser.parse_args()
    result = run_design_mode_alpha(
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
