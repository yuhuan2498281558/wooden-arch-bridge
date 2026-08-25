from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, HuberRegressor, QuantileRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVR

from ml_pipeline.prepare.symmetric_targets import (
    ASYMMETRY_PREPROCESSING,
    RATIO_DEFINITION,
    STRUCTURE_SCOPE_POLICY,
    TARGET_METHOD,
    ZERO_INNER_FLAT_CHORD_RATIO_THRESHOLD,
    collect_annotation_paths,
    prepare_annotation,
)


ARTIFACT_VERSION = "five-miao-node-pilot-v3"
FEATURE_NAMES = ("span_m",)
TARGETS = {
    "alpha": {"column": "design_target_alpha", "rule": 2.0 / 3.0, "bounds": (0.0, 1.0)},
    "beta": {"column": "design_target_beta", "rule": 1.0 / 4.0, "bounds": (0.0, 0.5)},
}
# v4 beta selected the lower edge (1e-4).  Extend the grid towards the
# unregularized limit so the next run can distinguish a real optimum from a
# boundary artefact while still keeping the high-regularisation candidates.
RIDGE_GRID = {"alpha": np.logspace(-8, 4, 25).tolist()}
HUBER_GRID = {
    "epsilon": [1.1, 1.2, 1.35, 1.5, 2.0],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
}
ELASTIC_NET_GRID = {
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
    "l1_ratio": [0.1, 0.5, 0.9],
}
SPLINE_RIDGE_GRID = {
    "n_knots": [3, 4, 5],
    "degree": [2, 3],
    "alpha": [1e-4, 1e-2, 1.0],
}
SVR_GRID = {
    "C": [0.5, 2.0, 8.0],
    "epsilon": [0.01, 0.04],
    "gamma": ["scale", 0.5],
}
GPR_GRID = {
    "length_scale": [0.5, 1.0, 2.0],
    "noise_level": [0.01, 0.05, 0.1],
}
EXTRA_TREES_GRID = {
    "max_depth": [2, 3, None],
    "min_samples_leaf": [2, 4, 8],
}
QUANTILE_GRID = {
    "alpha": [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
}
MODEL_SELECTION_TOLERANCE = 0.002
MODEL_SIMPLICITY_ORDER = (
    "fixed_rule",
    "train_median",
    "ridge",
    "huber",
    "quantile",
    "elastic_net",
    "spline_ridge",
    "svr",
    "gpr",
    "extra_trees",
)
DEFAULT_MODEL_NAMES = ("ridge", "huber")


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
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def load_training_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = [prepare_annotation(path) for path in paths]
    accepted: list[dict[str, Any]] = []
    for row in all_rows:
        exclusion_reasons: list[str] = []
        if row["needs_review"]:
            exclusion_reasons.append("needs_review")
        if row["quality"] == "low":
            exclusion_reasons.append("quality_low")
        if row.get("has_defects"):
            exclusion_reasons.append("defects")
        if row["training_scope_excluded"]:
            exclusion_reasons.extend(
                reason for reason in str(row["scope_exclusion_reasons"]).split(";") if reason
            )
        # Rows without a usable clear span cannot provide the design-time
        # feature; keep them traceable in prepared data but exclude them from
        # training instead of aborting the whole run (soft exclusion).
        try:
            span_m = float(row["span_m"])
        except (TypeError, ValueError):
            span_m = float("nan")
        if not math.isfinite(span_m) or span_m <= 0:
            exclusion_reasons.append("missing_span_m")
        row["training_exclusion_reasons"] = ";".join(exclusion_reasons)
        row["training_eligible"] = not exclusion_reasons
        if exclusion_reasons:
            continue
        accepted.append(row)

    # Enforce the project invariant that every bridge/span sample belongs to
    # exactly one split group and that sample keys are unique; otherwise the
    # leave-one-group-out guarantee silently degrades on inconsistent data.
    bridge_groups: dict[str, set[str]] = {}
    sample_keys: set[str] = set()
    for row in accepted:
        bridge_groups.setdefault(row["bridge_key"], set()).add(row["split_group_key"])
        if row["sample_key"] in sample_keys:
            raise ValueError(f"duplicate sample_key: {row['sample_key']}")
        sample_keys.add(row["sample_key"])
    inconsistent = {key: sorted(groups) for key, groups in bridge_groups.items() if len(groups) > 1}
    if inconsistent:
        raise ValueError(
            "bridge spans must not be split across split groups: "
            + "; ".join(f"{key}->{','.join(groups)}" for key, groups in sorted(inconsistent.items()))
        )

    group_count = len({row["split_group_key"] for row in accepted})
    if len(accepted) < 10 or group_count < 5:
        raise ValueError("pilot training requires at least 10 accepted rows from 5 groups")
    return all_rows, accepted


def feature_matrix(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def make_estimator(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "ridge":
        regressor = Ridge(alpha=float(params["alpha"]))
    elif model_name == "huber":
        regressor = HuberRegressor(
            alpha=float(params["alpha"]),
            epsilon=float(params["epsilon"]),
            max_iter=5000,
        )
    elif model_name == "elastic_net":
        regressor = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=10000,
            random_state=20260814,
        )
    elif model_name == "quantile":
        regressor = QuantileRegressor(
            quantile=0.5,
            alpha=float(params["alpha"]),
            solver="highs",
            fit_intercept=True,
        )
    elif model_name == "spline_ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            (
                "spline",
                SplineTransformer(
                    n_knots=int(params["n_knots"]),
                    degree=int(params["degree"]),
                    include_bias=False,
                ),
            ),
            ("regressor", Ridge(alpha=float(params["alpha"]))),
        ])
    elif model_name == "svr":
        regressor = SVR(
            kernel="rbf",
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
            gamma=params["gamma"],
        )
    elif model_name == "gpr":
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(float(params["length_scale"]), length_scale_bounds="fixed")
            + WhiteKernel(float(params["noise_level"]), noise_level_bounds="fixed")
        )
        regressor = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            optimizer=None,
            random_state=20260814,
        )
    elif model_name == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=1.0,
            random_state=20260814,
            n_jobs=1,
        )
        return Pipeline([("regressor", regressor)])
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline([("scale", StandardScaler()), ("regressor", regressor)])


def parameter_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "ridge":
        return list(ParameterGrid(RIDGE_GRID))
    if model_name == "huber":
        return list(ParameterGrid(HUBER_GRID))
    if model_name == "elastic_net":
        return list(ParameterGrid(ELASTIC_NET_GRID))
    if model_name == "spline_ridge":
        return list(ParameterGrid(SPLINE_RIDGE_GRID))
    if model_name == "svr":
        return list(ParameterGrid(SVR_GRID))
    if model_name == "gpr":
        return list(ParameterGrid(GPR_GRID))
    if model_name == "extra_trees":
        return list(ParameterGrid(EXTRA_TREES_GRID))
    if model_name == "quantile":
        return list(ParameterGrid(QUANTILE_GRID))
    raise ValueError(f"unknown model: {model_name}")


def tune_model(
    model_name: str,
    x: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    *,
    metric_groups: np.ndarray | None = None,
    inner_splits: int = 5,
) -> tuple[dict[str, Any], float]:
    unique_groups = np.unique(groups)
    folds = min(inner_splits, len(unique_groups))
    if folds < 2:
        raise ValueError("inner tuning requires at least two groups")
    splitter = GroupKFold(n_splits=folds)
    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    evaluation_groups = np.asarray(metric_groups if metric_groups is not None else groups)
    if len(evaluation_groups) != len(groups):
        raise ValueError("metric_groups must align with training rows")
    for params in parameter_grid(model_name):
        oof_prediction = np.full(len(residual), np.nan, dtype=float)
        failed = False
        for train_index, validation_index in splitter.split(x, residual, groups):
            estimator = make_estimator(model_name, params)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    estimator.fit(x[train_index], residual[train_index])
                oof_prediction[validation_index] = estimator.predict(x[validation_index])
            except (ValueError, FloatingPointError):
                failed = True
                break
        score = (
            float("inf")
            if failed or not np.all(np.isfinite(oof_prediction))
            else group_macro_mae(residual, oof_prediction, evaluation_groups)
        )
        if score < best_score:
            best_params, best_score = dict(params), score
    if best_params is None or not math.isfinite(best_score):
        raise RuntimeError(f"no valid {model_name} hyperparameters")
    return best_params, best_score


def group_macro_mae(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    values = [
        float(np.mean(np.abs(y_true[groups == group] - y_pred[groups == group])))
        for group in np.unique(groups)
    ]
    return float(np.mean(values))


def bootstrap_group_mae_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    random_seed: int,
) -> tuple[float, float]:
    unique_groups = np.unique(groups)
    group_errors = np.asarray([
        np.mean(np.abs(y_true[groups == group] - y_pred[groups == group])) for group in unique_groups
    ])
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(group_errors, size=(iterations, len(group_errors)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def metric_record(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_groups: np.ndarray,
    *,
    bridge_keys: np.ndarray | None = None,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    actual_bridge_keys = np.asarray(bridge_keys if bridge_keys is not None else split_groups)
    low, high = bootstrap_group_mae_interval(
        y_true, y_pred, actual_bridge_keys, iterations=bootstrap_iterations, random_seed=random_seed
    )
    bridge_mae = group_macro_mae(y_true, y_pred, actual_bridge_keys)
    split_group_mae = group_macro_mae(y_true, y_pred, split_groups)
    absolute_errors = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        # Backward-compatible alias.  From v5 onward it is explicitly the
        # true bridge-key macro score, not the leakage-prevention split group.
        "group_macro_mae": bridge_mae,
        "group_macro_mae_ci95": [low, high],
        "bridge_macro_mae": bridge_mae,
        "bridge_macro_mae_ci95": [low, high],
        "split_group_macro_mae": split_group_mae,
        "absolute_error_quantiles": {
            "q50": float(np.quantile(absolute_errors, 0.5)),
            "q90": float(np.quantile(absolute_errors, 0.9)),
            "q95": float(np.quantile(absolute_errors, 0.95)),
        },
        "max_absolute_error": float(np.max(absolute_errors)),
    }


def nested_group_evaluation(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    outer_splits: int | None = None,
    inner_splits: int = 5,
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260812,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    x = feature_matrix(rows, feature_names)
    groups = np.asarray([row["split_group_key"] for row in rows])
    bridge_keys = np.asarray([row["bridge_key"] for row in rows])
    if outer_splits is None:
        splitter = LeaveOneGroupOut()
    else:
        folds = min(int(outer_splits), len(np.unique(groups)))
        if folds < 2:
            raise ValueError("outer evaluation requires at least two groups")
        splitter = GroupKFold(n_splits=folds)
    evaluated_models = tuple(dict.fromkeys(("fixed_rule", "train_median", *model_names)))
    predictions: dict[str, dict[str, np.ndarray]] = {
        target: {name: np.zeros(len(rows), dtype=float) for name in evaluated_models}
        for target in TARGETS
    }
    tuning_history: dict[str, dict[str, list[dict[str, Any]]]] = {
        target: {name: [] for name in model_names} for target in TARGETS
    }

    for train_index, test_index in splitter.split(x, groups=groups):
        held_out_group = str(groups[test_index][0])
        train_groups = groups[train_index]
        for target_name, target_spec in TARGETS.items():
            y = np.asarray([float(row[target_spec["column"]]) for row in rows])
            rule = float(target_spec["rule"])
            predictions[target_name]["fixed_rule"][test_index] = rule
            predictions[target_name]["train_median"][test_index] = float(np.median(y[train_index]))
            residual = y[train_index] - rule
            for model_name in model_names:
                params, inner_mae = tune_model(
                    model_name,
                    x[train_index],
                    residual,
                    train_groups,
                    metric_groups=bridge_keys[train_index],
                    inner_splits=inner_splits,
                )
                estimator = make_estimator(model_name, params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    estimator.fit(x[train_index], residual)
                lower, upper = target_spec["bounds"]
                predictions[target_name][model_name][test_index] = np.clip(
                    rule + estimator.predict(x[test_index]), lower, upper
                )
                tuning_history[target_name][model_name].append({
                    "held_out_group": held_out_group,
                    "params": params,
                    "inner_bridge_macro_mae": inner_mae,
                    "inner_mae": inner_mae,
                })

    metrics: dict[str, Any] = {}
    for target_name, target_spec in TARGETS.items():
        y = np.asarray([float(row[target_spec["column"]]) for row in rows])
        metrics[target_name] = {
            model_name: metric_record(
                y,
                prediction,
                groups,
                bridge_keys=bridge_keys,
                bootstrap_iterations=bootstrap_iterations,
                random_seed=random_seed + index,
            )
            for index, (model_name, prediction) in enumerate(predictions[target_name].items())
        }

    oof_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        output = {
            "sample_key": row["sample_key"],
            "bridge_name": row["bridge_name"],
            "bridge_key": row["bridge_key"],
            "split_group_key": row["split_group_key"],
            "span_index": row["span_index"],
            "span_count": row["span_count"],
            "span_m": row["span_m"],
        }
        for target_name, target_spec in TARGETS.items():
            output[f"observed_{target_name}"] = row[target_spec["column"]]
            for model_name, prediction in predictions[target_name].items():
                output[f"predicted_{target_name}_{model_name}"] = float(prediction[index])
                output[f"absolute_error_{target_name}_{model_name}"] = abs(
                    float(row[target_spec["column"]]) - float(prediction[index])
                )
        oof_rows.append(output)
    return metrics, oof_rows, tuning_history


def choose_winner(
    target_metrics: dict[str, dict[str, float]],
    *,
    tolerance: float = MODEL_SELECTION_TOLERANCE,
) -> str:
    best_score = min(record.get("bridge_macro_mae", record["group_macro_mae"]) for record in target_metrics.values())
    eligible = {
        name for name, record in target_metrics.items()
        if record.get("bridge_macro_mae", record["group_macro_mae"]) <= best_score + tolerance
    }
    for model_name in MODEL_SIMPLICITY_ORDER:
        if model_name in eligible:
            return model_name
    raise ValueError("no eligible model")


def applicability_domain_record(
    x: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    nearest_neighbors: int = 5,
) -> dict[str, Any]:
    """Describe local training density without treating min/max as sufficient."""
    values = np.asarray(x, dtype=float)
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (values - center) / scale
    if len(normalized) < 2:
        distances = np.zeros(len(normalized), dtype=float)
        k = 0
    else:
        k = min(int(nearest_neighbors), len(normalized) - 1)
        pairwise = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        distances = np.mean(np.partition(pairwise, kth=k - 1, axis=1)[:, :k], axis=1)
    minimum_threshold = 0.35 * math.sqrt(values.shape[1])
    q95 = float(np.quantile(distances, 0.95)) if len(distances) else 0.0
    threshold = max(q95 * 1.5, minimum_threshold)
    return {
        "method": "standardized-mean-k-nearest-distance-v1",
        "feature_names": list(feature_names),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "normalized_training_points": normalized.tolist(),
        "nearest_neighbor_k": k,
        "training_distance_quantiles": {
            "q50": float(np.quantile(distances, 0.5)) if len(distances) else 0.0,
            "q90": float(np.quantile(distances, 0.9)) if len(distances) else 0.0,
            "q95": q95,
        },
        "sparse_region_threshold": threshold,
    }


def fit_artifact(
    target_name: str,
    model_name: str,
    rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_spec = TARGETS[target_name]
    x = feature_matrix(rows, feature_names)
    y = np.asarray([float(row[target_spec["column"]]) for row in rows])
    groups = np.asarray([row["split_group_key"] for row in rows])
    rule = float(target_spec["rule"])
    final_params: dict[str, Any] = {}
    estimator: Pipeline | None = None
    constant: float | None = None
    if model_name == "fixed_rule":
        constant = rule
    elif model_name == "train_median":
        constant = float(np.median(y))
    else:
        bridge_keys = np.asarray([row["bridge_key"] for row in rows])
        final_params, _ = tune_model(model_name, x, y - rule, groups, metric_groups=bridge_keys)
        estimator = make_estimator(model_name, final_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(x, y - rule)

    oof_errors = np.asarray([float(row[f"absolute_error_{target_name}_{model_name}"]) for row in oof_rows])
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "status": "pilot_not_for_production",
        "target": target_name,
        "target_column": target_spec["column"],
        "model_name": model_name,
        "feature_names": list(feature_names),
        "rule_baseline": rule,
        "constant": constant,
        "estimator": estimator,
        "bounds": list(target_spec["bounds"]),
        "training_feature_range": {
            name: [float(np.min(x[:, index])), float(np.max(x[:, index]))]
            for index, name in enumerate(feature_names)
        },
        "applicability_domain": applicability_domain_record(x, feature_names),
        "oof_absolute_error_quantiles": {
            "q50": float(np.quantile(oof_errors, 0.5)),
            "q90": float(np.quantile(oof_errors, 0.9)),
            "q95": float(np.quantile(oof_errors, 0.95)),
        },
        "coordinate_system": RATIO_DEFINITION,
        "target_method": TARGET_METHOD,
        "asymmetry_preprocessing": ASYMMETRY_PREPROCESSING,
        "structure_scope_policy": STRUCTURE_SCOPE_POLICY,
    }
    return artifact, final_params


def predict_artifact(artifact: dict[str, Any], features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if artifact.get("estimator") is None:
        prediction = np.full(features.shape[0], float(artifact["constant"]), dtype=float)
    else:
        prediction = float(artifact["rule_baseline"]) + artifact["estimator"].predict(features)
    lower, upper = artifact["bounds"]
    return np.clip(prediction, lower, upper)


def _save_plots(output_dir: Path, rows: list[dict[str, Any]], oof_rows: list[dict[str, Any]], winners: dict[str, str]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    span = np.asarray([float(row["span_m"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, (target_name, target_spec) in zip(axes, TARGETS.items()):
        observed = np.asarray([float(row[target_spec["column"]]) for row in rows])
        axis.scatter(span, observed, color="#1f5b8f", alpha=0.8)
        axis.set_xlabel("Clear span (m)")
        axis.set_ylabel(target_name)
        axis.set_title(f"Observed {target_name} vs span")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "targets_vs_span.png", dpi=180)
    plt.close(figure)

    for target_name in TARGETS:
        winner = winners[target_name]
        observed = np.asarray([float(row[f"observed_{target_name}"]) for row in oof_rows])
        predicted = np.asarray([float(row[f"predicted_{target_name}_{winner}"]) for row in oof_rows])
        low = min(float(np.min(observed)), float(np.min(predicted)))
        high = max(float(np.max(observed)), float(np.max(predicted)))
        figure, axis = plt.subplots(figsize=(5.5, 5))
        axis.scatter(observed, predicted, color="#ba4a32", alpha=0.82)
        axis.plot([low, high], [low, high], linestyle="--", color="#555")
        axis.set_xlabel(f"Observed {target_name}")
        axis.set_ylabel(f"OOF predicted {target_name}")
        axis.set_title(f"{target_name}: {winner} leave-one-split-group-out")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{target_name}_oof.png", dpi=180)
        plt.close(figure)


def _report_markdown(
    all_rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    metrics: dict[str, Any],
    winners: dict[str, str],
    final_params: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# 五节苗牛头节点比例 Pilot 训练报告",
        "",
        "> 状态：探索性模型，不得直接作为工程定型模型。",
        "",
        "## 数据",
        "",
        f"- 原始标注：{len(all_rows)} 条",
        f"- 参与训练：{len(accepted)} 条",
        f"- 独立分组：{len({row['split_group_key'] for row in accepted})} 组",
        f"- 待复核/排除：{len(all_rows) - len(accepted)} 条",
        f"- 零平弦特殊构造排除：{sum(bool(row['has_zero_inner_flat_chord']) for row in all_rows)} 条",
        f"- 保留并完成对称预处理的历史不对称观测：{sum(bool(row['has_observed_asymmetry']) for row in accepted)} 条",
        f"- 特征：{', '.join(FEATURE_NAMES)}",
        "- 外层验证：按 split_group_key 执行 Leave-One-Group-Out",
        "- 内层调参：按 split_group_key 分折，以真实 bridge_key 宏 MAE 评分",
        "",
        "## 桥级交叉验证结果",
        "",
        "| 目标 | 模型 | 样本 MAE | 桥级宏 MAE | 谱系分组宏 MAE | 95% CI | RMSE | R² |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target_name in TARGETS:
        for model_name, record in metrics[target_name].items():
            ci = record["group_macro_mae_ci95"]
            lines.append(
                f"| {target_name} | {model_name} | {record['mae']:.5f} | "
                f"{record['bridge_macro_mae']:.5f} | {record['split_group_macro_mae']:.5f} | "
                f"[{ci[0]:.5f}, {ci[1]:.5f}] | "
                f"{record['rmse']:.5f} | {record['r2']:.4f} |"
            )
    lines.extend(["", "## 选择结果", ""])
    for target_name in TARGETS:
        lines.append(
            f"- `{target_name}`：`{winners[target_name]}`；全量训练折最终参数 `{json.dumps(final_params[target_name], ensure_ascii=False)}`"
        )
    lines.extend([
        f"- 模型选择容差：桥级宏 MAE 与最优值相差不超过 {MODEL_SELECTION_TOLERANCE:.3f} 时，优先选择更简单模型。",
        "",
        "## 使用限制",
        "",
        "- 目前仅有净跨特征，桥宽、矢高、总长和节苗数量尚未进入训练表。",
        "- 数据来自单一数据库且没有独立外部测试集。",
        "- 标注者字段为空，尚不能估计标注者间误差。",
        "- 任何在线接入都必须保留可行域投影、训练范围检查和规则回退。",
        "",
    ])
    return "\n".join(lines)


def run_training(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260812,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    if not paths:
        raise ValueError(f"no annotation files matched {pattern!r} in {input_dir}")
    all_rows, accepted = load_training_rows(paths)
    metrics, oof_rows, tuning_history = nested_group_evaluation(
        accepted,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    winners = {target: choose_winner(metrics[target]) for target in TARGETS}
    final_params: dict[str, dict[str, Any]] = {}
    artifact_dir = output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for target_name, winner in winners.items():
        artifact, params = fit_artifact(target_name, winner, accepted, oof_rows)
        artifact["trained_at"] = datetime.now(timezone.utc).isoformat()
        artifact["training_rows"] = len(accepted)
        artifact["training_groups"] = len({row["split_group_key"] for row in accepted})
        artifact["cv_metrics"] = metrics[target_name][winner]
        joblib.dump(artifact, artifact_dir / f"{target_name}_model.joblib")
        final_params[target_name] = params

    excluded = [row for row in all_rows if row not in accepted]
    _write_csv(output_dir / "prepared_annotations.csv", all_rows)
    _write_csv(output_dir / "training_rows.csv", accepted)
    _write_csv(output_dir / "excluded_rows.csv", excluded)
    _write_csv(output_dir / "oof_predictions.csv", oof_rows)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "tuning_history.json", tuning_history)
    _write_json(output_dir / "model_selection.json", {"winners": winners, "final_params": final_params})

    try:
        import importlib.metadata as metadata

        package_versions = {
            name: metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib")
        }
    except Exception:
        package_versions = {}
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "status": "pilot_not_for_production",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "annotation_pattern": pattern,
        "source_files": [{"name": path.name, "sha256": _sha256(path)} for path in paths],
        "source_rows": len(all_rows),
        "training_rows": len(accepted),
        "training_groups": len({row["split_group_key"] for row in accepted}),
        "excluded_rows": len(excluded),
        "feature_names": list(FEATURE_NAMES),
        "targets": TARGETS,
        "target_method": TARGET_METHOD,
        "asymmetry_preprocessing": ASYMMETRY_PREPROCESSING,
        "structure_scope_policy": STRUCTURE_SCOPE_POLICY,
        "zero_inner_flat_chord_ratio_threshold": ZERO_INNER_FLAT_CHORD_RATIO_THRESHOLD,
        "coordinate_system": RATIO_DEFINITION,
        "validation": {
            "outer": "LeaveOneGroupOut",
            "inner": "GroupKFold(5)",
            "split_by": "split_group_key",
            "score_by": "bridge_key",
        },
        "selection_metric": "bridge_macro_mae",
        "selection_tolerance": MODEL_SELECTION_TOLERANCE,
        "model_simplicity_order": list(MODEL_SIMPLICITY_ORDER),
        "random_seed": random_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": package_versions,
        "git": _git_state(root),
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    _save_plots(output_dir, accepted, oof_rows, winners)
    (output_dir / "pilot_report.md").write_text(
        _report_markdown(all_rows, accepted, metrics, winners, final_params), encoding="utf-8"
    )
    return {"output_dir": str(output_dir), "winners": winners, "metrics": metrics, "final_params": final_params}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate pilot five-miao node ratio models.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260812)
    args = parser.parse_args()
    result = run_training(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
